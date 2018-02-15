<?php
$server = "http://127.0.0.1:5000";
$data = file_get_contents($server.$_SERVER["QUERY_STRING"]);
echo $data;
