import streamlit as st
import pandas as pd
import os
import tempfile
import matplotlib.pyplot as plt
import tempfile
from zipfile import ZipFile
from PIL import Image
import io

from SA import run_sa_from_files
from opt_script import run_two_opt_from_files
import random
random.seed(0)

st.set_page_config(page_title="📦 Route Optimizer", layout="centered")
st.title("📦 Warehouse Route Optimization")

method = st.selectbox("Select Optimization Method", ["Local Search", "Simulated Annealing"])

stok_file = st.file_uploader("📁 Upload Stock File", type="xlsx")
order_file = st.file_uploader("📁 Upload Order File", type="xlsx")
distance_file = st.file_uploader("📁 Upload Distance Matrix", type="xlsx")

if "output_df" not in st.session_state:
    st.session_state.output_df = None
if "excel_buffer" not in st.session_state:
    st.session_state.excel_buffer = None

if st.button("🚀 Run Optimization") and stok_file and order_file and distance_file:
    with st.spinner("Running optimization..."):
        with tempfile.NamedTemporaryFile(delete=False, suffix=".xlsx") as tmp_stok, \
             tempfile.NamedTemporaryFile(delete=False, suffix=".xlsx") as tmp_order, \
             tempfile.NamedTemporaryFile(delete=False, suffix=".xlsx") as tmp_dist:

            tmp_stok.write(stok_file.read())
            tmp_order.write(order_file.read())
            tmp_dist.write(distance_file.read())

            if method == "Simulated Annealing":
                output_df, excel_buffer = run_sa_from_files(tmp_stok.name, tmp_order.name, tmp_dist.name)
            else:
                output_df, excel_buffer = run_two_opt_from_files(tmp_stok.name, tmp_order.name, tmp_dist.name)

            st.session_state.output_df = output_df
            st.session_state.excel_buffer = excel_buffer

    st.success("✅ Optimization Complete!")

# Show core results
if st.session_state.output_df is not None and st.session_state.excel_buffer is not None:
    st.subheader("📍 Routes")
    st.dataframe(st.session_state.output_df)

    xls = pd.ExcelFile(st.session_state.excel_buffer)
    if "Remaining Stock" in xls.sheet_names:
        st.subheader("📦 Remaining Stock")
        st.dataframe(xls.parse("Remaining Stock"))

    st.download_button("📥 Download Full Excel", data=st.session_state.excel_buffer,
                       file_name=f"{method.replace(' ', '_')}_results.xlsx")
def get_coordinates(location):
    if location == "Initial Point":
        return (0, 0)
    x_values = {"11": 15.675, "12": 11.4, "13": 7.125, "14": 2.85}
    prefix = location[:2]
    if prefix in x_values:
        x = x_values[prefix]
        try:
            y = int(location[3:].split("-")[0])
        except:
            y = 0
        return (x, y)
    return (0, 0)

def create_route_plot(route_str, order_no, color_idx=0):
    route = route_str.split(" -> ")
    coords = [get_coordinates(loc) for loc in route]
    x_vals = [x for x, y in coords]
    y_vals = [y for x, y in coords]

    fig, ax = plt.subplots(figsize=(7, 5))
    cmap = plt.get_cmap("tab20")
    ax.plot(x_vals, y_vals, marker="o", color=cmap(color_idx % 20), linewidth=2)

    for i, (loc, (x, y)) in enumerate(zip(route, coords)):
        ax.text(x, y + 0.6, f"{i}. {loc}", fontsize=8, ha="center")

    ax.set_title(f"Order {order_no} Route")
    ax.set_xlabel("Aisle")
    ax.set_ylabel("Shelf")
    ax.grid(True)
    fig.tight_layout()
    return fig
if st.checkbox("📸 Show All Routes as Images + Download"):
    st.info("Rendering route plots...")

    with tempfile.TemporaryDirectory() as tmpdir:
        zip_path = os.path.join(tmpdir, "all_routes.zip")
        with ZipFile(zip_path, "w") as zipf:
            for idx, row in st.session_state.output_df.iterrows():
                route_str = row["Route"]
                order_no = row.get("Order No", row.get("Sipariş No", f"order_{idx}"))
                fig = create_route_plot(route_str, order_no, idx)

                # Display in app
                st.pyplot(fig)

                # Save to temp file
                img_bytes = io.BytesIO()
                fig.savefig(img_bytes, format="png", dpi=150)
                img_bytes.seek(0)
                img_name = f"order_{order_no}_route.png"
                zipf.writestr(img_name, img_bytes.read())
                plt.close(fig)

        # Show download button
        with open(zip_path, "rb") as f:
            st.download_button(
                label="⬇️ Download All Route Plots (ZIP)",
                data=f.read(),
                file_name="route_visualizations.zip",
                mime="application/zip"
            )
