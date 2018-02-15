<?php

use \Psr\Http\Message\ServerRequestInterface as Request;
use \Psr\Http\Message\ResponseInterface as Response;


$app->get('/scenario[/{id:[0-9]+}]', function (Request $request, Response $response, $args) {
    $pdo = $this->db;
    $stmt = ($args['id'] == null) ? $pdo->query('SELECT * FROM scenario') : $pdo->query("SELECT * FROM scenario where id={$args['id']}");
    $elements[]  = array();
    while ($row = $stmt->fetch()) {
        array_push($elements, $row);
    }
    array_shift($elements);
    echo json_encode($elements, JSON_PRETTY_PRINT);
    return $response->withHeader('Content-type','application/json');
});


$app->get('/scenario/time/', function (Request $request, Response $response, $args) {
    $pdo = $this->db;
    $query = "SELECT TIMESTAMPDIFF(SECOND,start_time,end_time) Duration FROM scenario";
    $stmt = $pdo->prepare($query);
    $stmt->execute();
    $elements[]  = array();
    echo json_encode($stmt->fetchAll(), JSON_PRETTY_PRINT);
    return $response->withHeader('Access-Control-Allow-Origin', '*')
        ->withHeader('Access-Control-Allow-Headers', 'X-Requested-With, Content-Type, Accept, Origin, Authorization')
        ->withHeader('Access-Control-Allow-Methods', 'GET, POST, PUT, DELETE, OPTIONS')
        ->withHeader('Content-type','application/json');
});


$app->get('/scenario/start_time={start_time}&end_time={end_time}', function (Request $request, Response $response, $args) {
    $pdo = $this->db;
    $query = "SELECT start_time,end_time FROM scenario WHERE start_time >= :start_time AND end_time <= :end_time";
    $stmt = $pdo->prepare($query);
    $stmt->execute(array(':start_time' => $args['start_time'], ':end_time' => $args['end_time']));
    $elements[]  = array();
    echo json_encode($stmt->fetchAll(), JSON_PRETTY_PRINT);
    return $response->withHeader('Access-Control-Allow-Origin', '*')
        ->withHeader('Access-Control-Allow-Headers', 'X-Requested-With, Content-Type, Accept, Origin, Authorization')
        ->withHeader('Access-Control-Allow-Methods', 'GET, POST, PUT, DELETE, OPTIONS')
        ->withHeader('Content-type','application/json');
});


$app->post('/scenario', function (Request $request, Response $response){
    $pdo = $this->db;
    $data = $request->getParsedBody();
    $query = "INSERT INTO scenario ( ";
    $params = array();
    foreach($data as $key => $value) {
        $query = $query." ".$key.", ";
        $params[':'.$key] = $value;
    }
    $query = substr($query, 0, -2).") VALUES ( ";
    foreach($data as $key => $value) {
        $query = $query." :".$key.", ";
    }
    $query = substr($query, 0, -2).")";
    $stmt = $pdo->prepare($query);
    $stmt->execute($params);
    return $response->withHeader('Content-type','application/json');
});


$app->put('/scenario/runid={runid}', function (Request $request, Response $response, $args) {
    $pdo = $this->db;
    $end_time = $request->getParsedBody()['end_time'];
    $stmt = $pdo->prepare("UPDATE scenario SET end_time = :end_time   where runid = :runid");
    $stmt->execute(array(':end_time' => $end_time, ':runid' => $args['runid']));
    return $response->withHeader('Content-type','application/json');
});

?>