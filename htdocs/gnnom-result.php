<?php
// error_reporting(E_ERROR | E_PARSE); // Suppresses warnings - e.g. when 'nan' is passed to number_format()

if($_SERVER['REQUEST_METHOD'] != 'POST') {  // Check if something was submitted here.
  header("Location: http://dara.embl-hamburg.de/");
  die;
}

require "dara.php";

$input_file           = $_FILES["userfile"]["name"];
$ang_units            = trim($_POST['ang_units']);
$nn_name              = trim($_POST['nn_name']);


$session_id = md5(uniqid (rand()));
$user_dir   = $user_dir_prefix . "gnnom/" . $session_id;
mkdir ($user_dir);
move_uploaded_file($_FILES["userfile"]["tmp_name"], "$user_dir/$input_file");
chdir ($user_dir) or die_message("could not change to directory {$user_dir}");

$input_file_info = pathinfo($input_file);

// Check if input file is a gnom.out file
/*
if(strtolower($input_file_info['extension']) == "out") {
  $input_gnom_file = $input_file;
  $input_dat_file  = $input_gnom_file . ".dat"; // TODO: $input_file_info['filename'] . ".dat"
  exec("out2fit " . $input_gnom_file . " > " . $input_dat_file);  // Creates a gnom.out.dat file with four columns which is used for crysol fitting and plotting; fourth column is ignored. The ".dat" extension is required by crysol.
}

else
*/

if(strtolower($input_file_info['extension']) == "dat") {
  $input_dat_file   = $input_file;
  // $input_gnom_file  = $input_file_info['filename'] . ".out";
  // $input_file_normed = $input_file_info['filename'] . "-normed.dat";
  $input_gnnom_file  = $input_file_info['filename'] . "-rebinned.dat";


  if($ang_units == 'unknown') {
    $returnvalue =
      s_range($input_dat_file, $smin, $smax);
    if($returnvalue != 0) {
      die_message("could not guess units from input data (s_range() returned {$returnvalue})");
    }
    if($smax < 1.0) {
      $ang_units = 'angstrom';
    } else {
      $ang_units = 'nanometer';
    }
  }
  
  
  // Estimate I(0) and Rg
  $returnvalue =
    autorg($input_dat_file, 10, 0.7, 1.1, $rg, $i0);

  if($returnvalue != 0) { // Assume the file is readable but no Rg found - run autorg with weakened conditions.
    $returnvalue =
      autorg($input_dat_file, 5, 1.3, 2.0, $rg, $i0);
    // echo "<br>Debug: Second AUTORG run";
  }

  if($returnvalue != 0) { // Assume the file is readable but no Rg found - run autorg with weakened conditions.
    $returnvalue =
      autorg($input_dat_file, 10, 0.7, 1.1, $rg, $i0, TRUE);
    // echo "<br>Debug: Third AUTORG run";
  }

  
  if($returnvalue != 0) {
    die_message("could not process input data (autorg() returned {$returnvalue}).");
  }
 
  // Normalize to I(0) = 1.0
  $returnvalue =
    datop($input_dat_file, "div", $i0, $input_gnnom_file);
  if($returnvalue != 0) {
    die_message("could not normalize input data (datop() returned {$returnvalue})");
  }  
  
    // Stretch rebinned to Rg = 20.0
  $returnvalue =
    datregrid_stretch($input_gnnom_file, $rg, $ang_units, "input_gnnom_stretched.dat"); 
  if($returnvalue != 0) {
    die_message("could not stretch input data (datregrid() returned {$returnvalue})");
  }
  
  // Rebin to correct grid
  $returnvalue =
    datregrid_scale($input_gnnom_file, $ang_units, $input_gnnom_file, "gnnom-input-template.dat"); 
  if($returnvalue != 0) {
    die_message("could not rebin input data (datregrid() returned {$returnvalue})");
  }
  
  // Rebin to correct grid
  $returnvalue =
    datregrid_scale("input_gnnom_stretched.dat", $ang_units, "input_gnnom_stretched.dat", "gnnom-input-template.dat"); 
  if($returnvalue != 0) {
    die_message("could not rebin input data (datregrid() returned {$returnvalue})");
  }
  // Datcrop to 4.5 nm 
    $smax = 0.45;
    $sfit = $smax * (20.0/$rg);
  if($ang_units == 'nanometer') {
    $smax = 4.5;
    $sfit = $smax * (2.0/$rg);
  }
  
  // Datcrop gnnom input to 4.5 nm 
  $returnvalue = datcrop_smax("input_gnnom_stretched.dat",$smax, "input_gnnom_stretched.dat");
  if($returnvalue != 0) {
    die_message("could not crop input_gnnom_stretched.dat (datcrop() returned {$returnvalue})");
  } 
  
  // Datcrop input to 4.5 * (20.0/Rg) nm 
  $returnvalue = datcrop_smax($input_dat_file, $sfit, $input_dat_file);
  if($returnvalue != 0) {
    die_message("could not crop input_gnnom_stretched.dat (datcrop() returned {$returnvalue})");
  } 
  
  // Check input data range
  $returnvalue =
    s_range($input_gnnom_file, $smin, $smax);
  if($returnvalue != 0) {
    die_message("could not interpret input data (s_range() returned {$returnvalue})");
  }
  
  // Check stretched input data range
  $returnvalue =
    s_range("input_gnnom_stretched.dat", $smin, $smax);
  if($returnvalue != 0) {
    die_message("could not interpret stretched input data (s_range() returned {$returnvalue})");
  }

  // TODO: Warn if Smin > 1.176470e-02 or if Smax < 4.509800e-01
  // if($smin > 0.0117647) {
    // TODO: extrapolate using $rg and I(0) = 1.0
    // die_message("s<sub>min</sub> = " . round($smin, 3) . " &gt; 0.012 &Aring;");
  // }

/*
  if($smin > 0.0117647) {
    echo "<strong>s<sub>min</sub> = " . round($smin, 3) . " &gt; 0.01 &Aring;</strong>";
  }
  if($smax < 0.45098) {
    echo "<strong>s<sub>max</sub> = " . round($smax, 3) . " &lt; 0.45 &Aring;</strong>";
  }
  
  echo "<br>Debug: Smin: " . $smin;
  echo "<br>Debug: Smax: " . $smax;
  echo "<br>Debug: Rg: "   . $rg;
  echo "<br>Debug: I(0): " . $i0;
  echo "<br>Debug: max Dmax: " . pi() / $smin . " = <strong>" . number_format(pi() / $smin / $rg, 1) . "*Rg</strong>";
  echo "<br>Debug: nn_name: " . $nn_name;
*/  

  // Predict Rg, Dmax, MW from *not stretched* input data
  $returnvalue =
    python_rgdmaxmw($input_gnnom_file, "gnnom_rg", $rg, $nn_rg);
  if($returnvalue != 0) {
    die_message("could not gnnom (python_rg() returned {$returnvalue})");
  }

  $returnvalue =
    python_rgdmaxmw($input_gnnom_file, "gnnom_dmax", $rg, $nn_dmax);
  if($returnvalue != 0) {
    die_message("could not gnnom (python_dmax() returned {$returnvalue})");
  }

  $returnvalue =
    python_rgdmaxmw($input_gnnom_file, "gnnom_mw", $rg, $nn_mw);
  if($returnvalue != 0) {
    die_message("could not gnnom (python_mw() returned {$returnvalue})");
  }
  
  // Re-normalize Rg and Dmax if needed
  $pscale = 1.0;
  if($ang_units == 'nanometer') {
    $pscale = 10.0;
  }
  
  // Run gnnom!
  $returnvalue =
    python_gnnom("input_gnnom_stretched.dat", $nn_name, "20.0"); // output file is always 'pddf-predicted.dat'
  if($returnvalue != 0) {
    die_message("could not gnnom (python_gnnom() returned {$returnvalue})");
  }
  
  // Squeeze p(r) back
  $returnvalue =
    datop("pddf-predicted.dat", "mul", $i0*(20.0/$rg), "pddf-predicted.dat");
  if($returnvalue != 0) {
    die_message("could not scale p(r) (datop() returned {$returnvalue})");
  } 

  // Convert predicted p(r) from A to nm if input was in 1/nm:
  $returnvalue =
    datregrid_scale("pddf-predicted.dat", $ang_units, "pddf-predicted.dat");
  if($returnvalue != 0) {
    die_message("could not scale p(r) (datregrid() returned {$returnvalue})");
  }

  // Stretch back p(r) to its original size
  $returnvalue =
    datregrid_stretch("pddf-predicted.dat", $rg, $ang_units, "pddf-predicted.dat"); 
  if($returnvalue != 0) {
    die_message("could not stretch input data (datregrid() returned {$returnvalue})");
  }

  $fit_file = $input_file_info['filename'] . ".fit";
  $cmd = "pddffit pddf-predicted.dat {$input_dat_file} --output {$fit_file}";
  exec($cmd);
  
  $returnvalue =
    datcmp_chi2($fit_file, $chi_sq);
  if($returnvalue != 0) {
    die_message("could not calculate chi^2 (gnnom_datcmp() returned {$returnvalue})");
  } 
  
  $returnvalue =
    datregrid_scale_back("pddf-predicted.dat", $ang_units, "pddf-predicted.dat"); 
  if($returnvalue != 0) {
    die_message("could not rebin input data (datregrid() returned {$returnvalue})");
  }

  $pddf_file = "pddf-predicted.dat";
  $pddf_rg   = get_pddf_rg($fit_file);
  $pddf_dmax = get_pddf_dmax($pddf_file);
}
else
  die_message("input file has unknown extension {$input_file_info['extension']}");




$title = $input_file;
include("header.php");
?>

<style type="text/css">
#resultstable {
  border-collapse: collapse;
  border: none;
  background: #ffffff;
}
#resultstable th, #resultstable td {
  border: 1px solid #f2f7fc;
  border-left-style:  none;
  border-right-style: none;
  padding: 15px;
  vertical-align: middle;
}

img.plot {
  border: 1px #cdd9ed;
  border-top-style:   none;
  border-right-style: none;
  border-bottom-style: dotted;
  border-left-style:   dotted;
}

div.alphabeta {
  width:100px;
  background-color: #F2F7FC;
  position:relative;
  margin-top: 15px;
}
div.alpha-bg {
  position: absolute;
  top: 0px;
  left: 0px;
  background-color:#c8c5ff;
}
div.beta-bg {
  position: absolute;
  top: 0px;
  right: 0px;
  background-color:#c9ffc7;
}
div.alpha-text {
  position: absolute;
  top: 0px;
  left: 2px;
  color:#342f91;
}
div.beta-text {
  position: absolute;
  top: 0px;
  right: 2px;
  color:#3eaf3b;
}
</style>



<h1>GNNOM</h1>




<?php
echo "<h3>" . $input_file_info['filename'] . "</h3>";

$unit = '&Aring;';
if($ang_units == 'nanometer') {
  $unit = 'nm';
}
echo "
<table>
  <tr>
    <td>Predicted R<sub>g</sub></td>
    <td>" . number_format($nn_rg / $pscale, 1) . "&nbsp;" . $unit . "</td>
  </tr>
  <tr>
    <td>p(r) R<sub>g</sub></td>
    <td>" . number_format($pddf_rg, 1) . "&nbsp;" . $unit . "</td>
  </tr>
  <tr>
    <td>autorg R<sub>g</sub></td>
    <td>" . number_format($rg, 1) . "&nbsp;" . $unit . "</td>
  </tr>
  <tr>
    <td>&nbsp;</td>
    <td>&nbsp;</td>
  </tr>
  <tr>
    <td>Predicted D<sub>max</sub></td>
    <td>" . number_format($nn_dmax / $pscale, 1) . "&nbsp;" . $unit . "</td>
  </tr>
  <tr>
    <td>p(r) D<sub>max</sub></td>
    <td>" . number_format($pddf_dmax, 1) . "&nbsp;" . $unit . "</td>
  </tr>
  <tr>
    <td>Predicted MW</td>
    <td>" . number_format($nn_mw, 1) . "&nbsp;kDa</td>
  </tr>
  <tr>
    <td>&#967<sup>2</sup></td>
    <td>" . number_format($chi_sq, 1) . "&nbsp;</td>
  </tr>    
</table>";

    // Plot the fit.
    // TODO: wrap this into a function?
    if($ang_units == "nanometer")
      $sscale = 1; // NB: DARA convention: 1/nm => 0.1; 1/A => 1.0.
    else
    $sscale = 10;

    $pngfitplot = $fit_file . ".png";
    $cmd = "gnuplot -e \"inputfile='"  . $fit_file .
               "'\" -e \"outputfile='" . $pngfitplot .
               "'\" -e sscale="        . $sscale .
                  " {$GLOBALS['gnnom_dir']}fit2img.cfg";
    exec($cmd);

    $pngresiduesplot = $fit_file . "-residues.png";
    $cmd = "gnuplot -e \"inputfile='"  . $fit_file .
               "'\" -e \"outputfile='" . $pngresiduesplot .
               "'\" -e sscale="        . $sscale .
                  " {$GLOBALS['gnnom_dir']}fitresiduals2img.cfg";
    exec($cmd);

    $pngpddfplot = $pddf_file . ".png";
    $cmd = "gnuplot -e \"inputfile='"  . $pddf_file .
               "'\" -e \"outputfile='" . $pngpddfplot .
               "'\" -e nn_dmax="        . $nn_dmax .
                  " {$GLOBALS['gnnom_dir']}pofr2img.cfg";
    exec($cmd);

  // echo $cmd; // debug
  echo "<br><img class='plot' src='./user/gnnom/{$session_id}/{$pngfitplot}' />&nbsp;<img class='plot' src='./user/gnnom/{$session_id}/{$pngpddfplot}' />";
  echo "<br><img class='plot' src='./user/gnnom/{$session_id}/{$pngresiduesplot}' />";
  // echo "<br>&chi;<sup>2</sup>: XXX, cormap p-value: " . $adj_Pr;
  echo "<br>";
  echo "<br>Download <a href='./user/gnnom/{$session_id}/{$fit_file}'>{$fit_file}</a>";
  echo "<br>Download <a href='./user/gnnom/{$session_id}/pddf-predicted.dat'>pddf-predicted.dat</a>";

  include("footer.php");
?>
