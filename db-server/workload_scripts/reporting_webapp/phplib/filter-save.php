<?php
    session_start();
    if (!isset($_SESSION['username'])) {
        header("Location: login.php");
    }

    $_SESSION['filters'] = $_POST;

    var_dump($_POST);
?>