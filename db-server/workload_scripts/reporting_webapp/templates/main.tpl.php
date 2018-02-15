<!doctype html>
<html lang="en">
    <head>
        <title><?php echo $PAGE_TITLE; ?></title>

        <!-- Required meta tags -->
        <meta charset="utf-8">
        <meta name="viewport" content="width=device-width, initial-scale=1, shrink-to-fit=no">
        <meta name="description" content="">
        <meta name="author" content="">

        <!-- Bootstrap CSS -->
        <link rel="stylesheet" href="css/lib/bootstrap/bootstrap.min.css">

        <!-- Other CSS -->
        <link rel="stylesheet" href="css/lib/datatables/jquery.dataTables.min.css"/>
        <link rel="stylesheet" href="css/lib/datatables/dataTables.bootstrap4.min.css"/>
        <link rel="stylesheet" href="css/lib/jqueryui/jquery-ui.css">

        <!-- Custom CSS -->
        <link rel="stylesheet" href="css/master.css">
        <?php
            if (!empty($CSS))
            {
                foreach ($CSS as $cssFileName)
                {
                    echo '<link rel="stylesheet" href="'.$cssFileName.'">';
                }
            }
            session_start();
        ?>
    </head>
    <body>
        <header>
            <nav class="navbar navbar-expand-md navbar-dark fixed-top bg-dark">
                <a class="navbar-brand" href="<?php echo $INDEX_FILE_NAME; ?>?l=<?php echo $PAGE_LANGUAGE;?>">
                    <?php echo $appName; ?>
                </a>
                <div class="collapse navbar-collapse">
                    <!--
                    <ul class="navbar-nav mr-auto">
                        <li class="nav-item active">
                            <a class="nav-link" href="#"><?php echo $homeLink; ?></a>
                        </li>
                        <li class="nav-item">
                            <a class="nav-link" href="#"><?php echo $settingsLink; ?></a>
                        </li>
                    </ul>
                    -->
                </div>
                <div class="hi">
                    <?php
                        if (isset($_SESSION['username'])) {
                            echo "Welcome, " . $_SESSION['username'];
                        }
                    ?>
                </div>
                <?php
                    if (isset($_SESSION['username'])) {
                        echo '<a class="nav-link" href="phplib/logout.php" >'.$logout.'</a>';
                    }
                ?>
                <div class="dropdown">
                    <button class="btn btn-secondary dropdown-toggle" type="button" id="menuLanguage" data-toggle="dropdown" aria-haspopup="true" aria-expanded="false">
                        <img src="images/choose-language.png"/>
                    </button>
                    <div class="dropdown-menu" aria-labelledby="menuLanguage">
                        <a class="dropdown-item int-en" href="#"><img src="images/gb.png"/></a>
                        <a class="dropdown-item int-ro" href="#"><img src="images/ro.png"/></a>
                    </div>
                </div>
            </nav>
        </header>

        <div class="container-fluid">
            <div class="row">
                <main role="main" class="col-sm-9 ml-sm-auto col-md-10 pt-3 mx-auto" id="main">
                </main>
            </div>
        </div>

        <!-- Placed at the end of the document so the pages load faster -->
        <!-- Libraries -->
        <!-- jQuery first, then Popper.js, then Bootstrap JS. Do not modify the loading order of these -->
        <script src="js/lib/jquery/jquery-3.2.1.min.js"></script>
        <script src="js/lib/popper/popper.min.js"></script>
        <script src="js/lib/bootstrap/bootstrap.min.js"></script>
            <!-- Charts http://www.chartjs.org/ -->
        <script src="js/lib/charts/Chart.bundle.min.js"></script>
            <!-- JQuery UI -->
        <script src="js/lib/jqueryui/jquery-ui.js"></script>
            <!-- DataTables -->
        <script src="js/lib/datatables/jquery.dataTables.min.js"></script>
        <script src="js/lib/datatables/dataTables.bootstrap4.min.js"></script>
        <script src="js/lib/datatables/dataTables.colReorder.min.js"></script>
        <script src="js/lib/datatables/dataTables.buttons.min.js"></script>

        <!--
        <script src="//cdnjs.cloudflare.com/ajax/libs/jszip/3.1.3/jszip.min.js"></script>
        <script src="//cdnjs.cloudflare.com/ajax/libs/pdfmake/0.1.32/pdfmake.min.js"></script>
        <script src="//cdnjs.cloudflare.com/ajax/libs/pdfmake/0.1.32/vfs_fonts.js"></script>
        <script src="//cdn.datatables.net/buttons/1.5.1/js/buttons.html5.min.js"></script>
        <script src="https://cdn.datatables.net/select/1.2.4/js/dataTables.select.min.js"></script>
        -->

        <!-- Own JS -->
        <script src="js/ajax.js" type="text/javascript"></script>
        <script>
            $(document).ready(function(){            
                var limba = "<?php echo $PAGE_LANGUAGE; ?>";
                var ip = "<?php echo $_SERVER['REMOTE_ADDR']; ?>";
                
                // completion calls
                <?php echo $AJAX_CONTENT['completionFunction']; echo "\n\n"; ?>
            
                // load the page content
                ajaxCall_generic("<?php echo $AJAX_CONTENT['page']; ?>", "<?php echo $AJAX_CONTENT['target']; ?>", <?php echo $AJAX_CONTENT['completionFunctionRunner']; ?>);

                // internationalization
                $(".int-en").click(function(){
                    <?php
                        $link = $_SERVER['SCRIPT_NAME'];
                        $i = 0;
                        $_GET["l"] = "en";
                        foreach ($_GET as $key => $value)
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
                        echo 'window.location = "'.$link.'"';
                    ?>
                });
                $(".int-ro").click(function(){
                    <?php
                        $link = $_SERVER['SCRIPT_NAME'];
                        $i = 0;
                        $_GET["l"] = "ro";
                        foreach ($_GET as $key => $value)
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
                        echo 'window.location = "'.$link.'"';
                    ?>
                });

            });
        </script>     
        <?php
            if (!empty($JS))
            {
                foreach ($JS as $jsFileName) 
                {               
                    echo '<script src="'.$jsFileName.'"></script>';
                }
            }
        ?>
    <script>
    </script>
    </body>
</html>