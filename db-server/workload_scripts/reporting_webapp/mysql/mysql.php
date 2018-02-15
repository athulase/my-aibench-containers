<?php
    $hostname_conn = "192.168.1.99";
    $database_conn = "dbreport";
    $username_conn = "root";
    $password_conn = "password";


    $conn = mysqli_connect($hostname_conn, $username_conn, $password_conn, $database_conn);
    if (mysqli_connect_errno()) {
        echo "Failed to connect to MySQL: " . mysqli_connect_error();
    }

?>
