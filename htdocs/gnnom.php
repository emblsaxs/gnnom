<?php
  $title = "gNNom";
  include("header.php");
?>

<script type="text/javascript">
  function CheckForm(form) {
    file_extension = form.userfile.value.split('.').pop().toLowerCase();
    if(file_extension == 'out' || file_extension == 'dat')
      return true;

    alert("Please provide a *.dat file or *.out GNOM file");
    form.userfile.focus();
    return false;
  }

  function handleFileSelect(evt) {
    var f = evt.target.files[0];
    var reader = new FileReader();

      reader.onload = function(e) {
          var span = document.getElementById('guess');
          var out = e.target.result;
          var i1 = out.lastIndexOf('Real space range');
          var i2 = out.lastIndexOf('Highest ALPHA (theor)') - i1;
          out = out.substr(i1, i2);
          i1  = out.indexOf('to') + 2;
          var a=parseFloat(out.substr(i1));
          if (a > 20.0) {
             var units= '1/&Aring;';
             span.innerHTML = "D<sub>max</sub>: "+a+", recommended units: "+units;
             document.getElementById("units").value = "angstrom";
          } else if  (a <= 20.0) {
             var units= '1/nm';
             span.innerHTML = "D<sub>max</sub>: "+a+", recommended units: "+units;
             document.getElementById("units").value = "nanometer";
          } // else
            // alert("Invalid GNOM file");
        };

      reader.readAsText(f);
  }
</script>

<h1>GNNOM</h1>
<h2>Fit a PDDF p(r) to experimental data</h2>
<br>
<form name="dara" id="dara" method="post" action="gnnom-result.php" enctype="multipart/form-data" onSubmit="return CheckForm(this);">
  <input type="hidden" name="n_number_total" value="100">
  <input type="hidden" name="sample_input" id="sample_input" value="">
  <table border="0" cellspacing="0" cellpadding="4">
    <tr>
      <td>Experimental data (*.dat)</td>
      <td><input type="file" name="userfile" id="gnomfile"><span id="guess"></span></td>
    </tr>
    <script type="text/javascript">
      document.getElementById('gnomfile').addEventListener('change', handleFileSelect, false);
    </script>
    <tr>
      <td>Angular units (optional)</td>
      <td>
        <select name="ang_units" id="units" style="width: 146px;" title="I'll guess the units if you don't specify them">
          <option value="unknown"></option>
          <option value="nanometer">1/nm</option>
          <option value="angstrom">1/&Aring;</option>
        </select> s = 4&pi;sin(&theta;)/&lambda;
      </td>
    </tr>
    <tr>
      <td>Macromolecule type</td>
      <td>
        <select name="nn_name" id="nn_name" style="width: 146px;" title="Name of the training data set">
          <option value="gnnom_pddf">protein</option>
        </select>
      </td>
    </tr>
    <tr>
      <td>&nbsp;</td>
      <td align="right"><input type="submit" value="Submit"></td>
    </tr>
  </table>
</form>

<?php include("footer.php"); ?>
