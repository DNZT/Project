import pandas as pd
import numpy as np
import random, math, time
from collections import defaultdict
from copy import deepcopy
from io import BytesIO
import random
random.seed(0)

def rename_input_columns(df):
    column_renames = {
        "Sipariş No": "Order No",
        "Gösterilen Sipariş No": "Order No",  # commonly appears this way
        "Ürün Kodu": "Product Code",
        "Toplanan Adet": "Picked Qty",
        "Lokasyon": "Location",
        "Son kullanma tarihi": "Expiration Date",
        "İç adet": "Available Qty",
        "Sipariş Koli sayısı": "Order Qty"
    }
    df.columns = df.columns.str.strip()
    df = df.rename(columns=lambda x: column_renames.get(x, x))
    return df
def edge_dist(a, b, matrix):
    try:
        return float(matrix.loc[a, b])
    except KeyError:
        try:
            return float(matrix.loc[b, a])
        except KeyError:
            return 0.0

def route_distance(route, matrix):
    return sum(edge_dist(route[i], route[i+1], matrix) for i in range(len(route)-1))

def total_distance(routes, matrix):
    return sum(route_distance(r, matrix) for r in routes.values())

def deduplicate_route(route):
    seen = set()
    inner = []
    for loc in route[1:-1]:
        if loc not in seen:
            inner.append(loc)
            seen.add(loc)
    return ["Initial Point", *inner, "Initial Point"]

def nearest_neighbor_route(locs, matrix):
    route = ["Initial Point"]
    remaining = set(locs)
    while remaining:
        current = route[-1]
        next_loc = min(
            remaining,
            key=lambda x: matrix.loc[current, x]
            if x in matrix.columns and current in matrix.index else float('inf')
        )
        route.append(next_loc)
        remaining.remove(next_loc)
    route.append("Initial Point")
    return route

def greedy_initial_solution(stok_df, sip_df, dist):
    stok_df["Son kullanma tarihi"] = pd.to_datetime(stok_df["Son kullanma tarihi"], dayfirst=True)
    stok_df = stok_df.sort_values(["ürün kodu", "Son kullanma tarihi"])
    remaining_stock = stok_df.copy()
    routes = {}
    picklists = defaultdict(list)
    collected_data = []

    for oid in sip_df["Gösterilen Sipariş No"].unique():
        rows = sip_df[sip_df["Gösterilen Sipariş No"] == oid]
        used_locs = set()

        for _, r in rows.iterrows():
            sku = r["ürün kodu"]
            need = r["Sipariş Koli sayısı"]
            stocks = remaining_stock[(remaining_stock["ürün kodu"] == sku) & (remaining_stock["İç adet"] > 0)]
            for idx, st in stocks.iterrows():
                loc = st["Lokasyon"]
                avail = st["İç adet"]
                take = min(avail, need)
                if take <= 0:
                    continue
                picklists[oid].append({
                    "SKU": sku,
                    "Lokasyon": loc,
                    "Adet": take,
                    "Tarih": st["Son kullanma tarihi"]
                })
                collected_data.append({
                    "Sipariş No": oid,
                    "Ürün Kodu": sku,
                    "Lokasyon": loc,
                    "Toplanan Adet": take
                })
                remaining_stock.at[idx, "İç adet"] -= take
                need -= take
                used_locs.add(loc)
                if need <= 0:
                    break

        valid_locs = [l for l in used_locs if l in dist.columns and l in dist.index]
        ruta = nearest_neighbor_route(valid_locs, dist)
        routes[oid] = deduplicate_route(ruta)

    return routes, picklists, remaining_stock, collected_data

def sa_global(routes0, map0, dist_matrix, T0, T_end=1.0, alpha=0.999, p_swap=0.1, sku_map=None):
    def make_2opt(routes):
        new_routes = deepcopy(routes)
        oid = random.choice(list(new_routes))
        r = new_routes[oid]
        if len(r) <= 3:
            return None, 0.0, [], "2-opt"
        i, j = sorted(random.sample(range(1, len(r)-1), 2))
        r[i:j+1] = reversed(r[i:j+1])
        new_routes[oid] = deduplicate_route(r)
        delta = route_distance(new_routes[oid], dist_matrix) - route_distance(routes[oid], dist_matrix)
        return new_routes, delta, [oid], "2-opt"

    def make_2swap(routes, mapping):
        sku = random.choice(list(sku_map))
        elig = [o for o in routes if sku in mapping[o]]
        if len(elig) < 2:
            return None, None, 0.0, [], "2-swap"
        o1, o2 = random.sample(elig, 2)
        l1, l2 = mapping[o1][sku], mapping[o2][sku]
        if l1 == l2:
            return None, None, 0.0, [], "2-swap"

        new_routes = deepcopy(routes)
        new_map = deepcopy(mapping)

        def replace(rt, old, new):
            for k in range(1, len(rt)-1):
                if rt[k] == old:
                    rt[k] = new
                    break

        replace(new_routes[o1], l1, l2)
        replace(new_routes[o2], l2, l1)
        new_routes[o1] = deduplicate_route(new_routes[o1])
        new_routes[o2] = deduplicate_route(new_routes[o2])
        new_map[o1][sku], new_map[o2][sku] = l2, l1

        delta = (
            route_distance(new_routes[o1], dist_matrix) - route_distance(routes[o1], dist_matrix)
            + route_distance(new_routes[o2], dist_matrix) - route_distance(routes[o2], dist_matrix)
        )
        return new_routes, new_map, delta, [o1, o2], "2-swap"

    best_routes = deepcopy(routes0)
    best_map = deepcopy(map0)
    best_dist = total_distance(best_routes, dist_matrix)
    cur_routes, cur_map = deepcopy(routes0), deepcopy(map0)
    cur_dist = best_dist
    T = T0
    logs = defaultdict(list)
    start_time = time.time()

    while T > T_end:
        if random.random() < p_swap:
            res = make_2swap(cur_routes, cur_map)
            if res[0] is None:
                T *= alpha
                continue
            new_routes, new_map, delta, affected, mtype = res
        else:
            new_routes, delta, affected, mtype = make_2opt(cur_routes)
            new_map = cur_map
            if new_routes is None:
                T *= alpha
                continue

        if delta < 0 or random.random() < math.exp(-delta / T):
            cur_routes, cur_map = new_routes, new_map
            cur_dist += delta
            now = time.time() - start_time
            for oid in affected:
                logs[oid].append({
                    "Time (s)": round(now, 2),
                    "Temp": round(T, 2),
                    "Move": mtype,
                    "Distance": route_distance(cur_routes[oid], dist_matrix),
                    "Route": " -> ".join(cur_routes[oid])
                })
            if cur_dist < best_dist:
                best_routes = deepcopy(cur_routes)
                best_map = deepcopy(cur_map)
                best_dist = cur_dist
        T *= alpha

    return best_routes, best_map, best_dist, logs

def run_sa_from_files(stock_path, order_path, distance_path):
    stok_df = pd.read_excel(stock_path)
    sip_df = pd.read_excel(order_path)

    dist = pd.read_excel(distance_path, index_col=0)
    dist.index = dist.index.astype(str).str.strip()
    dist.columns = dist.columns.astype(str).str.strip()

    routes, picklists, remaining_stock, collected_data = greedy_initial_solution(stok_df, sip_df, dist)

    sku_to_locs = stok_df.groupby("ürün kodu")["Lokasyon"].agg(set).to_dict()
    multi_loc_sku = {s: locs for s, locs in sku_to_locs.items() if len(locs) >= 2}

    order_sku_loc = defaultdict(dict)
    for o, plist in picklists.items():
        for rec in plist:
            order_sku_loc[o][rec["SKU"]] = rec["Lokasyon"]

    T0 = 50 * total_distance(routes, dist)
    final_routes, final_map, best_dist, logs = sa_global(
        routes, order_sku_loc, dist, T0, sku_map=multi_loc_sku
    )

    output_rows = [{
        "Sipariş No": o,
        "Route": " -> ".join(rt),
        "Distance": route_distance(rt, dist)
    } for o, rt in final_routes.items()]
    output_df = pd.DataFrame(output_rows)
    collected_df = pd.DataFrame(collected_data)
    output_df = output_df.rename(columns={
        "Sipariş No": "Order No",
        "Route": "Route",
        "Distance": "Distance"})  # or "Total Distance" → "Distance"
    remaining_stock = remaining_stock.rename(columns={
        "ürün kodu": "Product Code",
        "İç adet": "Available Qty",
        "Son kullanma tarihi": "Expiration Date",
        "Lokasyon": "Location"
    })

    collected_df = collected_df.rename(columns={
        "Sipariş No": "Order No",
        "Ürün Kodu": "Product Code",
        "Toplanan Adet": "Picked Qty",
        "Lokasyon": "Location"
    })

    buffer = BytesIO()
    with pd.ExcelWriter(buffer, engine="xlsxwriter") as writer:
        output_df.to_excel(writer, sheet_name="Optimized Routes", index=False)
        remaining_stock.to_excel(writer, sheet_name="Remaining Stock", index=False)
        collected_df.to_excel(writer, sheet_name="Collected Items", index=False)
        for oid, log_rows in logs.items():
            pd.DataFrame(log_rows).to_excel(writer, sheet_name=f"SA Log {str(oid)[:25]}", index=False)
    buffer.seek(0)

    return output_df, buffer
