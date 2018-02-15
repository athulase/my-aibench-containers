<?php
    require_once("language-detector.php");
    require_once("../language/".$language.".php");
    require_once("constants.php");
?>

<div class="container">
    <form class="form-signin" action="phplib/login.php" method="post">
        <h2 class="form-signin-heading"><?php echo $signIn; ?></h2>
        <label for="inputUsername" class="sr-only"><?php echo $username; ?></label>
        <input type="text" id="inputUsername" name="username" class="form-control" placeholder="Username" required autofocus>
        <label for="inputPassword" class="sr-only"><?php echo $password; ?></label>
        <input type="password" id="inputPassword" name="password" class="form-control" placeholder="Password" required>
        <div class="checkbox">
            <label>
                <input type="checkbox" value="remember-me"><?php echo $rememberMe; ?>
            </label>
        </div>
        <button class="btn btn-lg btn-primary btn-block" type="submit"><?php echo $signIn; ?></button>
    </form>
</div>



