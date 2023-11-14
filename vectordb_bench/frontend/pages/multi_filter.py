import streamlit as st
from vectordb_bench.frontend.components.check_results.headerIcon import drawHeaderIcon
from vectordb_bench.frontend.components.multi_filter.chart import drawCharts
from vectordb_bench.frontend.components.multi_filter.data import getChartsData
from vectordb_bench.frontend.components.multi_filter.dataFilter import dataFilter

from vectordb_bench.frontend.const.styles import FAVICON
from vectordb_bench.frontend.utils import getColor


def main():
    # set page config
    st.set_page_config(
        page_title="Multi-Filter Perf",
        page_icon=FAVICON,
        layout="wide",
        # initial_sidebar_state="collapsed",
    )

    # header
    drawHeaderIcon(st)

    chartsData = getChartsData()

    chartsData, selectedDbLabels = dataFilter(
        st.sidebar.container(),
        chartsData,
        header="DB",
        showLabel="dbLabel",
        defaultActivedWord="cardinal",
    )

    chartsData, _ = dataFilter(
        st.sidebar.container(),
        chartsData,
        header="Type",
        showLabel="filter_type",
        defaultActivedWord="",
    )

    chartsData, _ = dataFilter(
        st.sidebar.container(),
        chartsData,
        header="Clause Count",
        showLabel="clause_num",
        defaultActivedWord="1",
    )

    colorMap = {dbLabel: getColor(i) for i, dbLabel in enumerate(selectedDbLabels)}

    metrics = [
        "qps",
        # "serial_latency_p99",
        "recall",
    ]

    drawCharts(
        st,
        chartsData,
        colorMap=colorMap,
        showLabel="dbLabel",
        metrics=metrics,
        selectedDbLabels=selectedDbLabels,
    )


if __name__ == "__main__":
    main()
