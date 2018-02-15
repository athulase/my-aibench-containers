<?php

use \Psr\Http\Message\ServerRequestInterface as Request;
use \Psr\Http\Message\ResponseInterface as Response;


$app->get('/project[/{id:[0-9]+}]', function (Request $request, Response $response, $args) {
    $pdo = $this->db;
    $stmt = ($args['id'] == null) ? $pdo->query('SELECT * FROM project') : $pdo->query("SELECT * FROM project where id={$args['id']}");
    $elements[]  = array();
    while ($row = $stmt->fetch()) {
        array_push($elements, $row);
    }
    array_shift($elements);
    echo json_encode($elements, JSON_PRETTY_PRINT);
    return $response->withHeader('Content-type','application/json');
});

?>