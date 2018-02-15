<?php
    session_start();
    if (!isset($_SESSION['username'])) {
        header("Location: login.php");
    }

    require_once ('filter-execute.php');

    echo json_encode($filtered_data);
?>