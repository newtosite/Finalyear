

    <?php
  
	
        $filename = "rc.txt";
		if($handle=fopen($filename,'r')){
			while(!feof($handle)){
				$line = fgets($handle);
				echo $line;
				echo "<br>";
			}
        fclose($handle);
		}
    ?>
