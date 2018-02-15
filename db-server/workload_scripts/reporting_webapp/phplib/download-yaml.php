<?php
    session_start();
    if (!isset($_SESSION['username'])) {
        header("Location: login.php");
    }

    require_once ('filter-execute.php');

    $content = $_GET["content"];

    $timestamp = date_timestamp_get(date_create());
    $yaml_file_name = 'export/aibench-settings-'.$timestamp.'.yaml';
    
    file_put_contents('../'.$yaml_file_name, $content);
    

    header('Content-Type: application/octet-stream');
    header("Content-Disposition: attachment; filename='".$yaml_file_name."'");
    header('Content-Length: ' . filesize($yaml_file_name));
    header("Location: /".$yaml_file_name);
?>