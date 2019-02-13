

    <?php
  
	   error_reporting(E_ERROR|E_PARSE);
        $filename = "insu.txt";
		if (0 == filesize($filename)){
			echo "<h1 style = 'color:red;'><center><b>THE DOC HAS NOT YET BEEN SCANNED</b></center></h1>";
		}
		if($handle=fopen($filename,'r')){
			while(!feof($handle)){
				$line = fgets($handle);
				echo $line;
				echo "<br>";
			}
        fclose($handle);
		}else{
			echo "NO file<br>";
		}
    ?>
