import streamlit as st
from vectordb_bench.frontend.components.check_results.headerIcon import drawHeaderIcon
from vectordb_bench.frontend.components.single_filter.chart import (
    drawChartByFilterField,
)
from vectordb_bench.frontend.components.single_filter.data import getChartsData
from vectordb_bench.frontend.components.single_filter.dataFilter import dataFilter
from vectordb_bench.frontend.const.styles import FAVICON
from vectordb_bench.frontend.utils import getColor


def main():
    # set page config
    st.set_page_config(
        page_title="Single-Filter Perf",
        page_icon=FAVICON,
        layout="wide",
        # initial_sidebar_state="collapsed",
    )

    # header
    drawHeaderIcon(st)

    chartsData = getChartsData()

    showLabel = "dbLabel"
    defaultActivedWord = "cardinal"
    showData, selectedDbLabels = dataFilter(
        st.sidebar.container(),
        chartsData,
        showLabel=showLabel,
        defaultActivedWord=defaultActivedWord,
    )

    colorMap = {dbLabel: getColor(i) for i, dbLabel in enumerate(selectedDbLabels)}

    metrics = [
        "qps",
        # "serial_latency_p99",
        "recall",
    ]

    drawChartByFilterField(
        st, showData, colorMap=colorMap, showLabel=showLabel, metrics=metrics
    )


if __name__ == "__main__":
    main()
