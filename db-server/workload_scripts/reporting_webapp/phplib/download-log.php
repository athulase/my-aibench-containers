<?php
    session_start();
    if (!isset($_SESSION['username'])) {
        header("Location: login.php");
    }

    $the_folder = $_GET["path"];
    $timestamp = date_timestamp_get(date_create());
    $zip_file_name = 'export/log-'.$timestamp.'.zip';

    class FlxZipArchive extends ZipArchive {
        /** Add a Dir with Files and Subdirs to the archive
            @param string $location Real Location
            @param string $name Name in Archive
        **/
        public function addDir($location, $name) {
            $this->addEmptyDir($name);
             $this->addDirDo($location, $name);
        }

        /** Add Files & Dirs to archive
            @param string $location Real Location
            @param string $name Name in Archive
        **/
        private function addDirDo($location, $name) {
            $name .= '/';         $location .= '/';
            $dir = opendir ($location);
            while ($file = readdir($dir))    {
                if ($file == '.' || $file == '..') continue;
                $do = (filetype( $location . $file) == 'dir') ? 'addDir' : 'addFile';
                $this->$do($location . $file, $name . $file);
            }
        }
    }

    $za = new FlxZipArchive;
    $res = $za->open("../".$zip_file_name, ZipArchive::CREATE);
    if($res === TRUE)    {
        $za->addDir($the_folder, basename($the_folder));
        $za->close();
        header('Content-Type: application/zip');
        header("Content-Disposition: attachment; filename='".$zip_file_name."'");
        header('Content-Length: ' . filesize($zip_file_name));
        header("Location: /".$zip_file_name);
    } else {
        echo 'Could not create a zip archive';
    }
?>