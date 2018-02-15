<?php
use \Psr\Http\Message\ServerRequestInterface as Request;
use \Psr\Http\Message\ResponseInterface as Response;

require '../vendor/autoload.php';


$config['displayErrorDetails'] = true;
$config['addContentLengthHeader'] = false;

$config['db']['host']   = "192.168.1.99";
$config['db']['user']   = "root";
$config['db']['pass']   = "password";
$config['db']['dbname'] = "dbreport";

$app = new \Slim\App(["settings" => $config]);

$container = $app->getContainer();

$container['db'] = function ($c) {
    $db = $c['settings']['db'];
    $pdo = new PDO("mysql:host=" . $db['host'] . ";dbname=" . $db['dbname'], $db['user'], $db['pass']);
    $pdo->setAttribute(PDO::ATTR_ERRMODE, PDO::ERRMODE_EXCEPTION);
    $pdo->setAttribute(PDO::ATTR_DEFAULT_FETCH_MODE, PDO::FETCH_ASSOC);
    return $pdo;
};

$container['info'] = function ($c) {
    $db = $c['settings']['db'];
    $pdo = new PDO("mysql:host=" . $db['host'] . ";dbname=information_schema", $db['user'], $db['pass']);
    $pdo->setAttribute(PDO::ATTR_ERRMODE, PDO::ERRMODE_EXCEPTION);
    $pdo->setAttribute(PDO::ATTR_DEFAULT_FETCH_MODE, PDO::FETCH_ASSOC);
    return $pdo;
};

foreach (glob('*.php') as $filename){
    include_once $filename;
}

$app->get('/', function (Request $request, Response $response) {

    $pdo = $this->db;
    $info = $this->info;
    $stmt = $info->query("SELECT TABLE_NAME from information_schema.TABLES where TABLE_SCHEMA='dbreport'");
    $tables[]  = array();
    while ($row = $stmt->fetch()) {
        array_push($tables, $row["TABLE_NAME"]);
    }
    array_shift($tables);
    for($i=0; $i<count($tables); $i++){
        $elements[] = array();
        $st = $pdo -> query("SELECT * FROM ".$tables[$i]);
        while ($row = $st->fetch()) {
            array_push($elements, $row);
        }
        array_shift($elements);
        $tables[$i] = $elements;
    }

    echo json_encode($tables, JSON_PRETTY_PRINT);
    return $response->withHeader('Content-type','application/json');
});

$app->get('/filters', function (Request $request, Response $response, $args) {
    function call($table, $pdo){
        $stmt = $pdo->prepare("select COLUMN_NAME,COLUMN_TYPE from information_schema.columns where TABLE_NAME in ('".$table."')");
        $stmt->execute();
        $tables[] = array();
        $res = $stmt->fetchAll(PDO::FETCH_ASSOC);
        foreach ($res as $column){
            if (array_key_exists($table, $tables)){
                $data = array("column_name" => $column["COLUMN_NAME"], "column_type" => $column["COLUMN_TYPE"]);
                array_push($tables[$table], $data);
            } else {
                $tables[$table] = array();
            }
        }
        array_shift($tables);

        $elements [] = array();
        foreach ($tables as $table_name => $val){
            foreach ($val as $value){
                $query = 'Select Distinct (';
                $column_name = $value["column_name"];
                $query =  $query.$column_name.') from '.$table_name;
                $stmt = $pdo->prepare($query);
                $stmt->execute();
                while ($rows = $stmt->fetch()) {
                    foreach ($rows as $row){
                        if (array_key_exists($column_name, $elements)) {
                            array_push($elements[$column_name]["values"], $row);
                        } else {
                            $elements[$column_name] = array("values" => array($row), "table" => $table_name, "type" => $value["column_type"]);
                        }
                    }
                }

            }
        }
        array_shift($elements);
        return $elements;

    }
    $pdo = $this->db;
    $elements=array_merge(call("flow", $pdo),call("scenario", $pdo));
    echo json_encode($elements, JSON_PRETTY_PRINT);
    return $response->withHeader('Access-Control-Allow-Origin', '*')
                    ->withHeader('Access-Control-Allow-Headers', 'X-Requested-With, Content-Type, Accept, Origin, Authorization')
                    ->withHeader('Access-Control-Allow-Methods', 'GET, POST, PUT, DELETE, OPTIONS')
                    ->withHeader('Content-type','application/json');
});



$app->run();

?>