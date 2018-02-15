<?php

    require_once ('../mysql/mysql.php');

    function create_columns($tables, $conn, $duplicated_columns=array(), $excluded_columns=array("id")){
        $index = 0;
        $columns = array();
        foreach ($tables as $table){
            $query = "select COLUMN_NAME from information_schema.columns where TABLE_NAME in ('".$table."') order by COLUMN_NAME";
            if (!$result = $conn->query($query)) {
                return $columns;
            }
            $data = mysqli_fetch_all($result,MYSQLI_ASSOC);
            foreach($data as $column ){
                $name = $column["COLUMN_NAME"];
                if (!in_array($name, $excluded_columns)){
                    $as = $name;
                    $field = $name;
                    if (array_search($name, array_column($columns, 'field'))){
                        $field = $table.'_'.$name;
                        $as = $field;
                    }
                    array_push($columns, array('db' => substr($table,0,1).'.'.$name, 'dt' => $index, 'field' => $field, 'as' => $as, 'table' => $table));
                    $index++;
                    if(in_array($name, $duplicated_columns)){
                        array_push($columns, array('db' => substr($table,0,1).'.'.$name, 'dt' => $index, 'field' => $field, 'as' => $as, 'table' => $table));
                        $index++;
                    }
                }
            }
        }
        return $columns;
    }

    $duplicated_columns = array("project_id");
    $columns = create_columns(array("scenario", "flow"), $conn, $duplicated_columns);
?>