import plotly.express as px
from vectordb_bench.metric import metricUnitMap


def drawChartByFilterField(st, data, **kwargs):
    filter_fields = list({d["filter_field"] for d in data})
    filter_fields.sort(reverse=True)
    for filter_field in filter_fields:
        st.header(filter_field)
        filterData = [d for d in data if d["filter_field"] == filter_field]
        drawChartByDataset(st, filterData, **kwargs)


def drawChartByDataset(st, data, **kwargs):
    datasetList = list(set([d["dataset"] for d in data]))
    datasetList.sort()
    for dataset in datasetList:
        st.subheader(dataset)
        drawChartByMetric(st, [d for d in data if d["dataset"] == dataset], **kwargs)


def drawChartByMetric(st, data, metrics=["qps", "recall"], **kwargs):
    columns = st.columns(len(metrics))
    for i, metric in enumerate(metrics):
        container = columns[i]
        container.markdown(f"#### {metric}")
        drawChart(container, data, metric, **kwargs)


def getRange(metric, data, padding_multipliers):
    minV = min([d.get(metric, 0) for d in data])
    maxV = max([d.get(metric, 0) for d in data])
    padding = maxV - minV
    rangeV = [
        minV - padding * padding_multipliers[0],
        maxV + padding * padding_multipliers[1],
    ]
    return rangeV


def drawChart(st, data: list[object], metric, colorMap, showLabel):
    unit = metricUnitMap.get(metric, "")

    x = "filter_rate"
    xrange = getRange(x, data, [0.05, 0.1])

    y = metric
    yrange = getRange(y, data, [0.2, 0.1])

    data.sort(key=lambda a: a[x])

    fig = px.line(
        data,
        x=x,
        y=y,
        color=showLabel,
        text=metric,
        markers=True,
        color_discrete_map=colorMap,
    )
    fig.update_xaxes(range=xrange)
    fig.update_yaxes(range=yrange)
    fig.update_traces(textposition="bottom right", texttemplate="%{y:,.4~r}" + unit)
    fig.update_layout(
        margin=dict(l=0, r=0, t=40, b=0, pad=8),
        legend=dict(
            orientation="h", yanchor="bottom", y=1, xanchor="right", x=1, title=""
        ),
    )
    st.plotly_chart(fig, use_container_width=True)
