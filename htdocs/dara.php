<?php
$db_servername = "localhost";
$db_username   = "dara";
$db_password   = "1sxasoib";
$db_name       = "dara";

$user_dir_prefix      = "/var/www/dara/user/"; // TODO: use relative path
$dara_tree_dir_prefix = "/var/www/dara/querydara/kdtrees/";
$dara_tree_min_smax   = 0.0392157; // 1/A
$int_files_dir        = "/var/www/dara/querydara/intfiles/";
$gnnom_dir            = "/var/www/dara/querygnnom/";


// TODO: move stuff related to HTML human readable output to a separate file.
$nmr_download_prefix  = "/pdb-nmr/";
$nmr_download_suffix  = "_model1.pdb.gz";
$pdb_download_prefix  = "https://files.rcsb.org/download/";
$pdb_link_prefix      = "http://www.rcsb.org/pdb/explore/explore.do?structureId=";
$pdb_thunbnail_prefix = "http://www.rcsb.org/pdb/images/";
$pdb_thunbnail_suffix = "_bio_r_80.jpg?bioNum=";


function die_message($message) { // Should be called only before including header.php
  $title = "Error";
  include("header.php");
  echo "<p><strong>Error:</strong> ";
  echo $message;
  echo "</p>";
  echo "<button onclick='window.history.back()'>Go back</button>";
  include("footer.php");
  exit(1);
}


function pdb_link($name) {
  $pdbid       = substr($name, 0, 4);
  $pdbassembly = substr($name, 8, 2);
  if($pdbassembly < 2) {
    $title = strtoupper($pdbid) . " RCSB PDB page";
  }
  else {
    $title = strtoupper($pdbid) . " RCSB PDB page (biological assembly " . $pdbassembly . ")";
  }
  $a = "<a target='_blank' href='" . $GLOBALS['pdb_link_prefix'] . $pdbid . "' title='" . $title . "'>" . strtoupper($pdbid) . "<img src='/img/external-link-icon.png' alt='' /></a>";
  return $a;
}


function s_range($input_dat_file, &$smin, &$smax) {
  $lines = file($input_dat_file);
  $smin = -1;
  $smax = -1;

  // find Smin
  foreach ($lines as $line_num => $line) {
    $line_array = preg_split("/\s+/", $line, 3, PREG_SPLIT_NO_EMPTY);
    if(is_numeric($line_array[0]) && is_numeric($line_array[1])) {
      $smin = $line_array[0];
      break;
    }
  }
  if($smin < 0) {
    return -1;
  }

  // find Smax
  $lines = array_reverse($lines);
  foreach ($lines as $line_num => $line) {
    $line_array = preg_split("/\s+/", $line, 3, PREG_SPLIT_NO_EMPTY);
    if(is_numeric($line_array[0]) && is_numeric($line_array[1])) {
      $smax = $line_array[0];
      break;
    }
  }
  if($smin > $smax) {
    return -2;
  }
  return 0;
}


function get_pddf_rg($input_pddf_file) {
  $lines = array_reverse(file($input_pddf_file));
  foreach ($lines as $line_num => $line) {
    $line_array = explode(" ", $line);
    if($line_array[0] == "real-space-rg:") {
      $rg = $line_array[1];
      break;
    }
  }
  return $rg;
}


function get_pddf_dmax($input_pddf_file) {
  $lines = file($input_pddf_file);
  

  $lines = file($input_pddf_file);
  foreach ($lines as $line_num => $line) {
    $line_array = preg_split("/\s+/", $line, 3, PREG_SPLIT_NO_EMPTY);
    if(is_numeric($line_array[0]) && is_numeric($line_array[1])) {
      $dmax = $line_array[0];
    }
  }
  return $dmax;
}


function autorg($input_dat_file, $mininterval, $sminrg, $smaxrg, &$rg, &$i0, $is_lenient = FALSE) {
  $cmd = "autorg --format ssv --mininterval {$mininterval} --sminrg {$sminrg} --smaxrg {$smaxrg} {$input_dat_file}";
  if($is_lenient) {
    $cmd = $cmd . " --lenient";
  }
  exec(escapeshellcmd($cmd), $output, $returnvalue);
  $output_array = explode(" ", $output[0]);
  $rg = $output_array[0];
  $i0 = $output_array[2];
  return $returnvalue;
}


function datop($input_dat_file, $operator, $const, $output_dat_file) {
  $cmd = "datop {$operator} {$input_dat_file} {$const} --output {$output_dat_file}";
  exec(escapeshellcmd($cmd), $output, $returnvalue);
  return $returnvalue;
}

function datcrop_smax($input_dat_file, $smax,$output_dat_file) {
  $cmd = "datcrop {$input_dat_file} --smax {$smax} --output {$output_dat_file}";
  exec(escapeshellcmd($cmd), $output, $returnvalue);
  return $returnvalue;
}

function datregrid_scale($input_dat_file, $ang_units, $output_dat_file, $template = NULL) {
  $sscale = 1.0;
  if($ang_units == 'nanometer') {
    $sscale = 0.1;
  }

  $cmd = "datregrid {$input_dat_file} --scale {$sscale} --output {$output_dat_file}";
  exec(escapeshellcmd($cmd), $output, $returnvalue1);
  //echo "<br>Debug 1: " . $cmd;
  
  $returnvalue2 = 0;
  if($template != NULL) {
    $cmd = "datregrid {$output_dat_file} --template {$GLOBALS['gnnom_dir']}{$template} --output {$output_dat_file}"; // Careful with the relative path...
    exec(escapeshellcmd($cmd), $output, $returnvalue2);
    //echo "<br>Debug 2: " . $cmd;
  }

  return $returnvalue1 + $returnvalue2;
}

function datregrid_scale_back($input_dat_file, $ang_units, $output_dat_file) {
  $sscale = 0.1;
  if($ang_units == 'nanometer') {
    $sscale = 1.0;
  }

  $cmd = "datregrid {$input_dat_file} --scale {$sscale} --output {$output_dat_file}";
  exec(escapeshellcmd($cmd), $output, $returnvalue);
  //echo "<br>Debug 1: " . $cmd;
    return $returnvalue;
  }
  
function datregrid_stretch($input_dat_file, $rg, $ang_units, $output_dat_file) {
  $stretch = $rg / 20.0;
  if($ang_units == 'nanometer'){
  $stretch = $rg / 2.0;
  }
  $cmd = "datregrid {$input_dat_file} --scale {$stretch} --output {$output_dat_file}";
  exec(escapeshellcmd($cmd), $output, $returnvalue1);
  //echo "<br>Debug 3: " . $cmd;
  return $returnvalue1;
}

function datregrid_stretch_back($input_dat_file, $rg, $ang_units, $output_dat_file) {
  $stretch = 20.0 / $rg;
  if($ang_units == 'nanometer'){
  $stretch = 2.0 / $rg;
  }
  $cmd = "datregrid {$input_dat_file} --scale {$stretch} --output {$output_dat_file}";
  exec(escapeshellcmd($cmd), $output, $returnvalue1);
  //echo "<br>Debug 3: " . $cmd;
  return $returnvalue1;
}

function python_gnnom($input_dat_file, $nn_name = "gnnom", $guinier_rg) { // output file is always 'pddf-predicted.dat'
  $cmd = "{$GLOBALS['gnnom_dir']}gnnom.py {$GLOBALS['gnnom_dir']}/models/{$nn_name}.json {$GLOBALS['gnnom_dir']}/models/{$nn_name}.h5 {$input_dat_file} {$guinier_rg} > stdout.txt 2> stderr.txt";
  exec(escapeshellcmd($cmd), $output, $returnvalue);
  return $returnvalue;
}


function python_rgdmaxmw($input_dat_file, $nn_name, $guinier_rg, &$rgdmaxmw){
  $cmd = "{$GLOBALS['gnnom_dir']}gnnom.py {$GLOBALS['gnnom_dir']}/models/{$nn_name}.json {$GLOBALS['gnnom_dir']}/models/{$nn_name}.h5 {$input_dat_file} {$guinier_rg} 2> stderr.txt";
  exec(escapeshellcmd($cmd), $output, $returnvalue);
  // DEBUG
  //echo "<br>Debug: " . $cmd;
  //echo "<br>Debug: " . $nn_name;
  //echo "<br>Debug: " . $rgdmaxmw;
 
  $rgdmaxmw   = $output[count($output) - 1];

  return $returnvalue;
}


function gnom5_datcmp($input_dat_file, $dmax, $out_file, &$adj_Pr) {
  $cmd = "gnom --rmax {$dmax} --output {$out_file} {$input_dat_file}";
  exec(escapeshellcmd($cmd), $output, $returnvalue);
  if($returnvalue != 0) {
    $adj_Pr = 0;
    return $returnvalue;
  }

  unset($output);
  $cmd = "datcmp --format csv " . $out_file;
  exec(escapeshellcmd($cmd), $output, $returnvalue);
  $output_array = explode(",", $output[0]);
  $adj_Pr = array_pop($output_array);
  return $returnvalue;
}

function datcmp_chi2($fit_file, &$chi_sq) {
  $cmd = "datcmp --test=CHI-SQUARE --format CSV {$fit_file}";
  exec(escapeshellcmd($cmd), $output, $returnvalue);
  $output_array = explode(",", $output[0]);
  $chi_sq = ($output_array[3]);
  return $returnvalue;
}


function crysol_predict($input_file, $prefix) {
  $cmd = "crysol {$input_file} -p {$prefix} -lm 50 -sm 1.0 -ns 256"; // Same settings as used for generating the stored curves (takes ~16 times longer than default).
  // $cmd = "crysol {$input_file} -p {$prefix} -lm 20 -sm 0.501961 -ns 129"; // Same grid as stored curves but up to 0.5 1/A and 20 harmonics
  exec(escapeshellcmd($cmd), $output, $returnvalue);
  return $returnvalue;
}


function querydara($input_file_name, $ang_units, $type, $neighbours_num_total, &$output) {
  if ($type == 'p') {
    $dara_tree          = 'p_kdtreeRepo';
  }
  else if ($type == 'n') {
    $dara_tree          = 'n_kdtreeRepo';
  }
  else if ($type == 'np') {
    $dara_tree          = 'np_kdtreeRepo';
  }
  else {
    return -1;
  }

  $cmd = "querydara -n " . $neighbours_num_total .
              " --unit " . $ang_units .
                     " " . $input_file_name .
                     " " . $GLOBALS['dara_tree_dir_prefix'] . $dara_tree;
  exec(escapeshellcmd($cmd), $output, $returnvalue);

  return $returnvalue;
}


function dara_chi($input_file_name, $ang_units, $neighbours, &$pdbid_chi, &$pdbid_multiplier) {
  // dara_chilst requires a file (int.lst) with a list of neighbours' curves (*.int):
  $intlistfile = fopen("int.lst", "w");
  if(!$intlistfile)
    return 14;
  foreach($neighbours as $name) {
    $int_folder = substr($name, 1 ,2);
    fwrite($intlistfile, $GLOBALS['int_files_dir'] . $int_folder . "/" . $name . ".int\n");
  }
  fclose($intlistfile);

  $cmd = "dara_chilst " . $input_file_name . " int.lst " . strtoupper($ang_units);
  exec(escapeshellcmd($cmd), $output, $returnvalue);
  if($returnvalue != 0)
    return $returnvalue;

  unlink("int.lst");

  // dara_chilst prints chi-square, multiplier and int file separated by tabs.
  // The order of the output is the same as in the int.lst file.
  // If dara_chilst fails it prints "nan" instead of chi-square and multiplier.
  if(is_numeric(strtok($output[0], "\t"))) {
    foreach($neighbours as $i => $name) {
      $pdbid_chi[        $name ] = strtok($output[$i], "\t");
      $pdbid_multiplier[ $name ] = strtok("\t");
    }
  }
  else {
    foreach($neighbours as $i => $name) {
      $pdbid_chi[        $name ] = 100.0;
      $pdbid_multiplier[ $name ] = 1.0;
    }
  }
  return 0;
}


function dara_lsqrfactor($input_file_name, $ang_units, $neighbours, &$pdbid_chi, &$pdbid_multiplier) {
  // $ang_units is ignored for now, theoretical data should be in 1/A.
  foreach($neighbours as $name) {
    $int_folder = substr($name, 1 ,2);
    $cmd = "dara_lsqrfactor " . $input_file_name .
                          " " . $GLOBALS['int_files_dir'] . $int_folder . "/" . $name . ".int";

    $dara_chi_output = preg_split("/\s+/", exec(escapeshellcmd($cmd))); // Last line with lsqDaraRF

    if(is_numeric($dara_chi_output[0])) {
      $pdbid_chi[        $name ] = $dara_chi_output[0];
      $pdbid_multiplier[ $name ] = $dara_chi_output[2];
    }
    else {
      $pdbid_chi[        $name ] = 100.0;
      $pdbid_multiplier[ $name ] = 1.0;
    }
  }
  return 0;
}


// TODO: join query_db() and query_clusters_db()
function query_db($pdbid_chi, &$rows) {
  // A way to use prepare() to bind multiple values to a single parameter in the IN() clause of an SQL statement:
  $inQuery = implode(',', array_fill(0, count(array_keys($pdbid_chi)), '?'));

  try {
    $conn = new PDO("mysql:host={$GLOBALS['db_servername']};dbname={$GLOBALS['db_name']}", $GLOBALS['db_username'], $GLOBALS['db_password']);
    $conn->setAttribute(PDO::ATTR_ERRMODE, PDO::ERRMODE_EXCEPTION);

    $stmt = $conn->prepare("SELECT id, alpha, beta, rg, dmax, mw, volume_excluded, volume_shell FROM parameters WHERE id IN (" . $inQuery . ")");

    $stmt->execute(array_keys($pdbid_chi));

    $rows = $stmt->fetchAll(PDO::FETCH_GROUP);
  }
  catch(PDOException $e) {
    echo "Error: " . $e->getMessage(); // TODO: do not print the error message here
    return $e->getCode(); // This might be not an integer?
  }
  $conn = null;
  return 0;
}


function query_clusters_db($pdbid_chi, &$rows) {
  // A way to use prepare() to bind multiple values to a single parameter in the IN() clause of an SQL statement:
  $inQuery = implode(',', array_fill(0, count(array_keys($pdbid_chi)), '?'));

  try {
    $conn = new PDO("mysql:host={$GLOBALS['db_servername']};dbname={$GLOBALS['db_name']}", $GLOBALS['db_username'], $GLOBALS['db_password']);
    $conn->setAttribute(PDO::ATTR_ERRMODE, PDO::ERRMODE_EXCEPTION);
    $stmt = $conn->prepare("SELECT representative, member FROM clusters WHERE representative IN (" . $inQuery . ")");
    $stmt->execute(array_keys($pdbid_chi));

    $rows = $stmt->fetchAll(PDO::FETCH_GROUP);
  }
  catch(PDOException $e) {
    echo "Error: " . $e->getMessage(); // TODO: do not print the error message here
    return $e->getCode(); // This might be not an integer?
  }
  $conn = null;
  return 0;
}

?>
