$(document).ready(function() {
    /*******************************************************************************************************************
     * Constants
     ******************************************************************************************************************/
    var link = "api.php?";

    /*******************************************************************************************************************
     * Aiding functions
     ******************************************************************************************************************/
    var formatDate = function(date) {
        var day = date.getDate();
        var month = date.getMonth() + 1;
        var year = date.getFullYear();
        var hour = date.getHours();
        var minutes = date.getMinutes();
        var seconds = date.getSeconds();
        return month + "/" + day + "/" + year + " " + hour + ":" + minutes + ":" + seconds;
    };

    var entityMap = {
        '&': '&amp;',
        '<': '&lt;',
        '>': '&gt;',
        '"': '&quot;',
        "'": '&#39;',
        '/': '&#x2F;',
        '`': '&#x60;',
        '=': '&#x3D;'
    };

    var escapeHtml = function(string) {
        return String(string).replace(/[&<>"'`=\/]/g, function (s) {
            return entityMap[s];
        });
    };

    /*******************************************************************************************************************
     * UI
     ******************************************************************************************************************/
    var createFilteringDate = function() {
        function getDate( element ) {
            var date;
            try {
                date = $.datepicker.parseDate(dateFormat, element.value);
            } catch( error ) {
                date = null;
            }
            return date;
        }
        var dateFormat = "mm/dd/yy", from, to;
        from = $("#from").datepicker({
            defaultDate: "+1w",
            changeMonth: true,
            changeYear: true
        }).on( "change", function() {
            to.datepicker( "option", "minDate", getDate(this) );
        });
        to = $("#to").datepicker({
            defaultDate: "+1w",
            changeMonth: true,
            changeYear: true
        }).on( "change", function() {
            from.datepicker( "option", "maxDate", getDate(this) );
        });
    };

    var createCharts = function() {
        $.get("phplib/filter-table-show.php", function(rawdata){
            var chartsContainer = $("#charts");
            chartsContainer.html("");
            var data = JSON.parse(rawdata)["data"];
            var chartData = [];
            var lookup = function(obj, lkey) {
                var value = "";
                $.each(dbcolumns, function(i, col){
                    if (col["field"] == lkey) {
                        value = obj[i];
                    }
                });
                return value;
            };
            $.each(data, function(i, e){
                var name = lookup(e, "scenario_name");
                var throughput_value = lookup(e, "throughput_value");
                var throughput_meaning = lookup(e, "throughput_meaning");
                var start = new Date(lookup(e, "start_time"));
                var obj = {};
                $.each(chartData, function(j, elem){
                    if (elem["label"] == name) {
                        obj = elem;
                    }
                });
                if ($.isEmptyObject(obj)) {
                    obj["label"] = name;
                    obj["data"] = [];
                    obj["plot_labels"] = [];
                    chartData.push(obj);
                }
                obj["data"].push(throughput_value);
                obj["data_meaning"] = throughput_meaning;
                obj["plot_labels"].push(formatDate(start));
            });
            Chart.defaults.global.defaultFontColor = "#000000";
            $.each(chartData, function(i, e){
                chartsContainer.append('<canvas class="chart" id="chart-' + i + '"></canvas>');
                var ctx = document.getElementById("chart-" + i).getContext("2d");
                var throughput_meaning = e["data_meaning"];
                var myChart = new Chart(ctx, {
                    type:'line',
                    options: {
                        scales: {
                            yAxes: [{
                                scaleLabel: {
                                    display:true,
                                    labelString:"Throughput ("+throughput_meaning+")"
                                }
                            }],
                            xAxes:[{
                                scaleLabel:{
                                    display:true,
                                    labelString:"Start time"
                                }
                            }]
                        },
                        legend: {
                            display: true
                        },
                        tooltips: {
                            callbacks: {
                                label: function(tooltipItem, data){
                                    return data.datasets[0].data[tooltipItem.index] + " " + throughput_meaning;
                                }
                            }
                        }
                    },
                    data: {
                        labels: e["plot_labels"],
                        datasets:[e]
                    }
                });
            });
        });
    };

    /*******************************************************************************************************************
     * Main zone
     ******************************************************************************************************************/
    var main = function() {
        createFilteringDate();
        createCharts();
        $.get(link + "/filters",function (data) {
            var columns = JSON.parse(data);

            /* Push the columns to UI */
            var excludedColumns = ["id", "runid", "project_id", "running_configuration", "logs_location"];
            var pushFilterElement = function(columnName, columnData) {
                var table = columnData["table"];
                var row = $('<div class="col-md-4 filteringDiv"></div>').appendTo($("#filter-panel-content"));
                var label = $('<label class="filteringLabel" for=' + columnName + '>'+columnLabels[columnName] + '</label>').appendTo(row);
                var select = $('<select data-table="'+table+'" class="filteringSelect" id =' + columnName + '><option value=""></option></select>').appendTo(row);
                $.each(columnData["values"], function(i, e){
                    select.append('<option value="' + e + '">' + e + '</option>');
                });
            };
            $.each(columns, function (columnName, columnData) {
                if ($.inArray(columnName, excludedColumns) == -1
                    && columnData["type"] != 'datetime') {
                    pushFilterElement(columnName, columnData);
                }
            });

            // Filter Button Action
            $("#filter-panel-apply").click(function(){
                var data = {};
                $(".filteringSelect").each(function(i,e){
                    var column = $(e).attr("id");
                    var value = {
                        "table": $(e).attr("data-table"),
                        "value": $(e).val()
                    };
                    data[column] = value;
                });
                var startDate = $("#from").datepicker('getDate');
                var endDate = $("#to").datepicker('getDate');
                data["start_time"] = {
                    "table": "scenario",
                    "value": startDate == null ? "" : startDate.getTime()/1000
                };
                data["end_time"] = {
                    "table": "scenario",
                    "value": endDate == null ? "" : endDate.getTime()/1000
                };
                $.post("phplib/filter-save.php", data, function(r){
                    $("#table-row-details").html("");
                    table.ajax.reload();
                    createCharts();
                });
            });

            $("#filter-panel-reset").click(function() {
                $.post("phplib/filter-save.php", {}, function(r){
                    $("#table-row-details").html("");
                    table.ajax.reload();
                    createCharts();
                });
            });

            // Export table data as CSV
            $("#filter-table-csv").click(function(){
                window.open("phplib/download-csv.php");
            });

            // Draw the table on screen
            var visibleColumns = ["scenario_name", "throughput_value", "throughput_meaning", "start_time", "cpu", "ml_framework", "model", "dataset", "running_node"];
            var dateColumns = ["start_time"];
            var visibleIndex = [];
            var invisibleIndex = [];
            var dateIndex = [];
            $.each(dbcolumns, function(i, dbc){
                var found = false;
                $.each(visibleColumns, function(j, e){
                    if (e == dbc["field"]) {
                        found = true;
                        visibleIndex.push(i);
                    }
                });
                $.each(dateColumns, function(j, e){
                    if (e == dbc["field"]) {
                        dateIndex.push(j);
                    }
                });
                if (!found) {
                    invisibleIndex.push(i);
                }
            });
            table = $('#filter-table').DataTable( {
                "serverSide":true,
                "processing":true,
                "ajax": "phplib/filter-table-show.php",
                dom: 'Bfrtip',
                buttons:[],
                select: true,
                searching: false,
                columnDefs:[
                    {
                        "visible":true,
                        "targets":visibleIndex
                    },
                    {
                        "visible":false,
                        "targets":invisibleIndex
                    },
                    {
                        "render": function(data, type, row, meta) {
                            return formatDate(new Date(data));
                        },
                        "targets": dateColumns
                    }
                ]
            });

            // Table click
            $('#filter-table tbody').on( 'click', 'tr', function () {
                $("#table-row-details").html("");
                var rowData = table .row(this).data();
                var i = 0;
                table.columns().every(function () {
                    var name = this.header().className.split(' ')[0];
                    if (name != "sorting") {
                        var aux = "";
                        if (name == "logs_location") {
                            aux = '<span class="input-group-btn"><button id="log-dl" class="btn btn-secondary" type="button">Download</button></span>';
                        }
                        if (name == "running_configuration") {
                            //aux = '<span class="input-group-btn"><button id="conf-dl" class="btn btn-secondary" type="button">Download</button></span>';
                        }
                        var displayed = escapeHtml(rowData[i]);
                        $("#table-row-details").append('<div class="row"><div class="col-md-12"><div class="input-group input-group-sm"><span class="input-group-addon detailsLabel">'+columnLabels[name]+'</span><input type="text" class="form-control detailsData" value="'+displayed+'" />'+aux+'</div></div></div>');
                    }
                    i++;
                });
                $("#log-dl").click(function(){
                    window.open("phplib/download-log.php?path="+$(this).parent().prev().val());
                });
                $("#conf-dl").click(function(){
                    window.open("phplib/download-yaml.php?content="+$(this).parent().prev().val());
                });
            });

        });
    };
    main();
});

