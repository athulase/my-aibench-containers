<?php
    session_start();
    if (!isset($_SESSION['username'])) {
        header("Location: login.php");
    }

    require_once ('filter-execute.php');

    $timestamp = date_timestamp_get(date_create());
    $csv_file_name = 'export/csv-export-'.$timestamp.'.csv';
    $fp = fopen('../'.$csv_file_name, 'w');

    $header = array();
    foreach ($filtered_data["columns"] as $column) {
        array_push($header, $column["field"]);
    }
    fputcsv($fp, $header);
    foreach ($filtered_data["data"] as $fields) {
        fputcsv($fp, $fields);
    }
    fclose($fp);

    header('Content-Type: application/csv');
    header("Content-Disposition: attachment; filename='".$csv_file_name."'");
    header('Content-Length: ' . filesize($csv_file_name));
    header("Location: /".$csv_file_name);
?>