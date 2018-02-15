<?php
    require_once("language-detector.php");
    require_once("../language/".$language.".php");
    require_once("constants.php");

    session_start();
    require_once ('../mysql/mysql.php');
    if($_SERVER["REQUEST_METHOD"] == "POST") {
        // username and password sent from form
        $myusername = mysqli_real_escape_string($conn,$_POST['username']);
        $mypassword = mysqli_real_escape_string($conn,$_POST['password']);
        $hashed_password = hash("sha256", $mypassword);

        $sql = "SELECT id FROM user WHERE username = ? and password = ?";
        $stmt = $conn->prepare($sql);
        $stmt->bind_param('ss',$myusername, $hashed_password);
        $stmt->execute();

        $result =$stmt->get_result();
       // $row = mysqli_fetch_array($result,MYSQLI_ASSOC);

        $count = $result->num_rows;
        $stmt->close();

        if($count == 1) {
            $_SESSION['username'] = $myusername;
            $_SESSION['password'] = $hashed_password;


            header("location: ../index.php");

        }else {

           header("location: ../login.php");


        }
    }
?>