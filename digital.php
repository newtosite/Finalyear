<?php
if($fh=fopen("license.txt",'r'))
{
	while(!feof($fh))
	{
		$line=fgets($fh);
		echo $line;
		echo"<br>";
	}
	fclose($fh);
}
?>