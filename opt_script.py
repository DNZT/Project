import pandas as pd
from io import BytesIO
import random
random.seed(0)

def calculate_total_distance(route, df_distance):
    total = 0
    for i in range(len(route) - 1):
        try:
            total += df_distance.loc[route[i], route[i + 1]]
        except KeyError:
            try:
                total += df_distance.loc[route[i + 1], route[i]]
            except KeyError:
                total += 0
    return total

def remove_duplicates(route):
    if not route:
        return ["Initial Point", "Initial Point"]
    clean_route = []
    seen = set()
    for loc in route:
        if loc == "Initial Point" and (not clean_route or clean_route[-1] != "Initial Point"):
            clean_route.append(loc)
        elif loc != "Initial Point" and loc not in seen:
            clean_route.append(loc)
            seen.add(loc)
    if clean_route[-1] != "Initial Point":
        clean_route.append("Initial Point")
    if clean_route[0] != "Initial Point":
        clean_route = ["Initial Point"] + clean_route
    return clean_route

def optimize_single_route(route, df_distance):
    route = remove_duplicates(route)
    improved = True
    history = []
    iteration = 0
    while improved:
        improved = False
        best_distance = calculate_total_distance(route, df_distance)
        core_route = route[1:-1]
        for i in range(len(core_route) - 1):
            for j in range(i + 1, len(core_route)):
                new_core = core_route[:i] + core_route[i:j+1][::-1] + core_route[j+1:]
                new_route = ["Initial Point"] + new_core + ["Initial Point"]
                new_route = remove_duplicates(new_route)
                new_distance = calculate_total_distance(new_route, df_distance)
                if new_distance < best_distance:
                    route = new_route
                    best_distance = new_distance
                    improved = True
                    iteration += 1
                    history.append({
                        "Iteration": iteration,
                        "Route": " -> ".join(route),
                        "Distance": round(best_distance, 2)
                    })
                    break
            if improved:
                break
    return route, best_distance, history

def greedy_initial_solution(stok_verileri, siparis_verileri, df_distance):
    stok_verileri["Son kullanma tarihi"] = pd.to_datetime(stok_verileri["Son kullanma tarihi"], dayfirst=True)
    stok_verileri = stok_verileri.sort_values(by=["ürün kodu", "Son kullanma tarihi"])
    initial_solution = []
    remaining_stock = stok_verileri.copy()
    collected_summary = []

    for siparis_no in siparis_verileri["Gösterilen Sipariş No"].unique():
        order = siparis_verileri[siparis_verileri["Gösterilen Sipariş No"] == siparis_no]
        pick_locations = set()
        for _, row in order.iterrows():
            urun = row["ürün kodu"]
            miktar = row["Sipariş Koli sayısı"]
            stoklar = remaining_stock[(remaining_stock["ürün kodu"] == urun) & (remaining_stock["İç adet"] > 0)]
            for idx, stok in stoklar.iterrows():
                lokasyon = stok["Lokasyon"]
                mevcut = stok["İç adet"]
                alinan = min(mevcut, miktar)
                if alinan > 0:
                    remaining_stock.at[idx, "İç adet"] -= alinan
                    miktar -= alinan
                    pick_locations.add(lokasyon)
                    collected_summary.append({
                        "Sipariş No": siparis_no,
                        "Ürün Kodu": urun,
                        "Lokasyon": lokasyon,
                        "Toplanan Adet": alinan
                    })
                if miktar <= 0:
                    break
        used_locations = sorted([loc for loc in pick_locations if loc in df_distance.columns and loc in df_distance.index])
        rota = ["Initial Point"] + used_locations + ["Initial Point"]
        rota = remove_duplicates(rota)
        initial_solution.append({"Sipariş No": siparis_no, "Rota": rota})

    return initial_solution, remaining_stock, collected_summary

def run_two_opt_from_files(stock_path, order_path, distance_path):
    random.seed(0)

    stok_verileri = pd.read_excel(stock_path)
    siparis_verileri = pd.read_excel(order_path)
    df_distance = pd.read_excel(distance_path, index_col=0)
    df_distance.index = df_distance.index.astype(str).str.strip()
    df_distance.columns = df_distance.columns.astype(str).str.strip()

    initial_solution, final_stock, collected_data = greedy_initial_solution(stok_verileri, siparis_verileri, df_distance)

    results = []
    all_history = []

    for entry in initial_solution:
        sip_no = entry["Sipariş No"]
        rota = entry["Rota"]
        optimized_route, dist, history = optimize_single_route(rota, df_distance)
        results.append({
            "Sipariş No": sip_no,
            "Route": " -> ".join(optimized_route),
            "Distance": round(dist, 2)
        })
        for h in history:
            h["Sipariş No"] = sip_no
            all_history.append(h)

    output_df = pd.DataFrame(results)
    history_df = pd.DataFrame(all_history)
    collected_df = pd.DataFrame(collected_data)
    output_df = output_df.rename(columns={
        "Sipariş No": "Order No",
        "Route": "Route",
        "Distance": "Distance"})  # or "Total Distance" → "Distance"
    final_stock = final_stock.rename(columns={
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



    # Only output_df and final_stock will be shown on Streamlit
    buffer = BytesIO()
    with pd.ExcelWriter(buffer, engine="xlsxwriter") as writer:
        output_df.to_excel(writer, sheet_name="Optimized Routes", index=False)
        final_stock.to_excel(writer, sheet_name="Remaining Stock", index=False)
        history_df.to_excel(writer, sheet_name="2-Opt History", index=False)
        collected_df.to_excel(writer, sheet_name="Collected Items", index=False)
    buffer.seek(0)

    return output_df, buffer
