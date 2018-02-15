<?php

    require_once ('../mysql/mysql.php');
    require_once ('columns.php');
    require('ssp.class.php');

    $sql_details = array(
        'user' => $username_conn,
        'pass' => $password_conn,
        'db'   => $database_conn,
        'host' => $hostname_conn
    );

    $table = 'scenario';

    $primaryKey = 'id';

    $joinQuery = "FROM scenario AS  s LEFT JOIN flow AS f ON (s.runid = f.runid)";
    $filtered_data = SSP::simple( $_GET, $sql_details, $table, $primaryKey, $columns, $joinQuery);
?>