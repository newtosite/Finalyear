<?php
error_reporting(0);
//$keyword = json_decode($_POST['query']);
$lat = json_decode($_POST['current_lat']);
$lng = json_decode($_POST['current_lng']);

$token_url = "https://outpost.mapmyindia.com/api/security/oauth/token?grant_type=client_credentials";

$access_token="";
$token_type="";

$curl_token = curl_init();
curl_setopt($curl_token, CURLOPT_URL, $token_url);
curl_setopt($curl_token, CURLOPT_POST, 1);
curl_setopt($curl_token, CURLOPT_RETURNTRANSFER, 1);
curl_setopt($curl_token, CURLOPT_SSL_VERIFYHOST, 0);
curl_setopt($curl_token, CURLOPT_SSL_VERIFYPEER, 0);
curl_setopt($curl_token, CURLOPT_POSTFIELDS,
            "client_id=mJWqVn7cnU9bdsprVliFOnWHHKsJZ-mW&client_secret=oL71ebD-EiTqUqyf8hLu4X-mxPxe3lSZysjOA1PnMlI=");
$result_token = curl_exec($curl_token);
$json = json_decode($result_token, true);
$access_token = $json['access_token'];
$token_type = $json['token_type'];
curl_close($curl_token);

$url="";
if($lat!="" && $lng!="")
{
	$url="https://atlas.mapmyindia.com/api/places/nearby/json?keywords=temple;petrol&refLocation=".$lat.",".$lng."";
}

$header = array();
$header[] = 'Content-length: 0';
$header[] = 'Content-type: application/json';
$header[] = 'Authorization: '.$token_type.' '.$access_token.'';

$curl = curl_init();
curl_setopt($curl, CURLOPT_URL, $url);
curl_setopt($curl, CURLOPT_RETURNTRANSFER, 1);
curl_setopt($curl, CURLOPT_SSL_VERIFYHOST, 0);
curl_setopt($curl, CURLOPT_SSL_VERIFYPEER, 0);
curl_setopt($curl, CURLOPT_VERBOSE, 1);
curl_setopt($curl, CURLOPT_HEADER, 1);
curl_setopt($curl, CURLOPT_CUSTOMREQUEST, "GET");
curl_setopt($curl, CURLOPT_HTTPHEADER, $header);
$result = curl_exec($curl);
$http_status = curl_getinfo($curl, CURLINFO_HTTP_CODE);

$header_size = curl_getinfo($curl, CURLINFO_HEADER_SIZE);
$response_header[] =explode("\r\n", substr($result, 0, $header_size));
$body[] = substr($result, $header_size);

curl_close($curl);

if($http_status=='200')
{
	$res['status']='success';
    $res['data']=$body;
    echo json_encode($res);
}
elseif($http_status=='400'){
    
    $res['status']='fail';
    $res['data']="No result found";
    echo json_encode($res);
}
else{

	$res['status']='fail';
    $res['data']=str_replace("message:", "", $response_header[0][6]);
    echo json_encode($res);
}
?>