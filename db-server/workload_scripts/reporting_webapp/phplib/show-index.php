<?php
    require_once("language-detector.php");
    require_once("../language/".$language.".php");
    require_once("constants.php");
    require_once("columns.php");

    $colLabels  = array(
        "cpu"                   => $columnLabels[0],
        "dataset"               => $columnLabels[1],
        "docker_image_link"     => $columnLabels[2],
        "start_time"            => $columnLabels[3],
        "end_time"              => $columnLabels[4],
        "ml_framework"          => $columnLabels[5],
        "ml_framework_compiler" => $columnLabels[6],
        "ml_framework_version"  => $columnLabels[7],
        "model"                 => $columnLabels[8],
        "scenario_name"         => $columnLabels[9],
        "flow_name"             => $columnLabels[10],
        "runid"                 => $columnLabels[11],
        "running_node"          => $columnLabels[12],
        "commitid"              => $columnLabels[13],
        "logs_location"         => $columnLabels[14],
        "project_id"            => $columnLabels[15],
        "running_configuration" => $columnLabels[16],
        "version"               => $columnLabels[17],
        "flow_start_time"       => $columnLabels[19],
        "flow_end_time"         => $columnLabels[20],
        "throughput_value"      => $columnLabels[21],
        "throughput_meaning"    => $columnLabels[22]
    );
?>

<script>
    var columnLabels  = <?php echo json_encode($colLabels); ?>;
    var dbcolumns = <?php echo json_encode($columns); ?>
</script>

<script src="js/index.js"></script>

<div id="accordion" role="tablist" aria-multiselectable="true">
  <div class="card">
    <div class="card-header" role="tab" id="filter-pane-title">
      <h5 class="mb-0">
        <a data-toggle="collapse" data-parent="#accordion" href="#collapseOne" aria-expanded="true" aria-controls="collapseOne">
          <?php echo $filters; ?>
        </a>
      </h5>
    </div>
    <div id="collapseOne" class="collapse" role="tabpanel" aria-labelledby="filter-pane-title">
      <div id="filter-panel-content" class="card-block">
        <div class="col-md-4 filteringDiv">
            <label for="from"><?php echo $from; ?></label>
            <input type="text" id="from" name="from" >
            <label for="to"><?php echo $to; ?></label>
            <input type="text" id="to" name="to">
        </div>
      </div>
      <div id="filter-panel-buttons">
        <button id="filter-panel-reset" class="btn btn-primary"><?php echo $reset; ?></button>
        <button id="filter-panel-apply" class="btn btn-primary"><?php echo $apply; ?></button>
    </div>
    </div>
  </div>
</div>

<br>
<br>


<ul class="nav nav-tabs" role="tablist">
    <li class="nav-item">
        <a class="nav-link active" data-toggle="tab" href="#table-view" role="tab"><?php echo $tableView; ?></a>
    </li>
    <li class="nav-item">
        <a class="nav-link" data-toggle="tab" href="#chart-view" role="tab"><?php echo $chartView; ?></a>
    </li>
</ul>
<div class="tab-content">
    <div class="tab-pane active" id="table-view" role="tabpanel">
        <div class="row container col-md-12">
            <button id="filter-table-csv" class="btn btn-primary"><?php echo $exportCsv; ?></button>
        </div>
        <div class="row container col-md-12">
            <table id="filter-table" class="table table-striped table-bordered table-sm" cellspacing="0" width="100%">
                <thead>
                    <tr>
                        <?php
                             foreach ($columns as $column) {
                                 $field = $column['field'];
                                 echo '<th class="'.$field.'">'.$colLabels[$field].'</th>';
                             }
                        ?>
                    </tr>
                </thead>
            </table>
        </div>
        <div class="row container col-md-12">
            <div class="col-md-12">
                <div class="card-header">
                   <?php echo $details; ?>
                </div>
                <div class="card-block">
                    <div id="table-row-details"></div>
                </div>
            </div>
        </div>
    </div>
    <div class="tab-pane" id="chart-view" role="tabpanel">
        <div id="charts"></div>
    </div>
</div>
