<?php
require_once("phplib/constants.php");
require_once("phplib/language-detector.php");
require_once("language/".$language.".php");

session_start();
if (!isset($_SESSION['username'])) {
    header("Location: login.php");
}
//else {
//
//    echo $_SESSION['username'];
//
//}

$link = "phplib/show-index.php";
$i = 0;
foreach ($_GET as $key => $value)
{
    {
        if ($i==0)
        {
            $link = $link."?".$key."=".$value;
            $i=1;
        }
        else
        {
            $link = $link."&".$key."=".$value;
        }
    }
}

$PAGE_LANGUAGE = $language;

$PAGE_TITLE = "AIBench Reporting Dashboard";

$CSS = array("css/index.css");

$JS = array();

$AJAX_CONTENT = array(
    "page" => $link,
    "completionFunction" => "var afterPageLoad = function(response){
                                
                            }",
    "completionFunctionRunner" => "afterPageLoad",
    "target" => "#main"
);

require_once("templates/main.tpl.php");
?>