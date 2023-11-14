
def dataFilter(
    st, data, showLabel="dbName", defaultActivedWord="", header="DB"
):
    st.subheader(header)
    dbLabels = list({d[showLabel] for d in data})
    dbLabels.sort()
    # dbLabels.remove("")
    dbLabelIsActived = {
        label: True if defaultActivedWord in label.lower() else False
        for label in dbLabels
    }
    for dbLabel in dbLabels:
        dbLabelIsActived[dbLabel] = st.checkbox(
            dbLabel, value=dbLabelIsActived[dbLabel]
        )

    def getShowData(data):
        return [d for d in data if dbLabelIsActived[d[showLabel]]]

    showData = getShowData(data)

    selectedDbLabels = [label for label in dbLabels if dbLabelIsActived[label]]

    return showData, selectedDbLabels
