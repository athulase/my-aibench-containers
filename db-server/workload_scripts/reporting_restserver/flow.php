<?php

use \Psr\Http\Message\ServerRequestInterface as Request;
use \Psr\Http\Message\ResponseInterface as Response;


$app->get('/flow[/{id:[0-9]+}]', function (Request $request, Response $response, $args) {
    $pdo = $this->db;
    $stmt = ($args['id'] == null) ? $pdo->query('SELECT * FROM flow') : $pdo->query("SELECT * FROM flow where id={$args['id']}");
    $elements[]  = array();
    while ($row = $stmt->fetch()) {
        array_push($elements, $row);
    }
    array_shift($elements);
    echo json_encode($elements, JSON_PRETTY_PRINT);
    return $response->withHeader('Content-type','application/json');
});


$app->put('/flow/runid={runid}', function (Request $request, Response $response, $args){
    $pdo = $this->db;
    $data = $request->getParsedBody();
    $query = "UPDATE flow SET ";
    $params = array(':runid' => $args['runid']);
    foreach($data as $key => $value) {
        $query = $query." ".$key."=".":".$key.", ";
        $params[':'.$key] = $value;
    }
    $query = substr($query, 0, -2)." WHERE runid=:runid";
    $stmt = $pdo->prepare($query);
    $stmt->execute($params);
    return $response->withHeader('Content-type','application/json');
});


$app->post('/flow', function (Request $request, Response $response){
    $pdo = $this->db;
    $data = $request->getParsedBody();
    $query = "INSERT INTO flow ( ";
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

?>