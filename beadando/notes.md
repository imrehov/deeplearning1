# Plans
elsőkörben eldöntöttem, hogy egy pythonscript fogja orchesztrálni az egész folyamatot, valószínűleg két megközelítésben vizsgálom meg a datasetet, egy MLP-vel fogok bináris klasszifikációt vagy attack klasszifikációt készíteni, illetve egy autoencodert fogok megvalósítani
az adatokat a ToN-IOT datasetből nyertem, ami kutatási és oktatási célra szabadon felhasználható insert link

a következő lépés, hogy a datasetből érdemes lenne elhagyni bizonyos oszlopokat, mert data leakage-t okozhatnak, konkrétan szerepelnek az egyes támadások típusai, illetve sokszor egy ip egy adott támadásnál jelenik csak meg

tehát a python script fogja kezelni a fileokat, segítségével készítek majd ideiglenes fájlmásolatokat amiből a hálózatok tanulnak
drop:
src_ip
dst_ip

dns_query

ssl_version
ssl_cipher
ssl_resumed
ssl_established
ssl_subject
ssl_issuer

http_trans_depth
http_method
http_uri
http_version
http_request_body_len
http_response_body_len
http_status_code
http_user_agent
http_orig_mime_types
http_resp_mime_types

weird_name
weird_addl
weird_notice

type

az MLP-k a teljes datasetet megkapják, az autencoder csak a normal (label == 0) rekordokat kapja meg


# execution

## coldrops
droppoltam az oszlopokat

## train/validation/test split
szétszedem a datasetet három különböző részre 70-15-15 split

log1p az összes dataseten
a standardscaler kiszámolja a szórást és átlagokat a tanításhoz használt dataseten
majd ugyanezzel a scalerrel transzformálom az 9sszest datasetet 

## onehot
most a nem szám oszlopokat one-hot encodeolni kellene, ezek: "proto", "conn_state", "service", "dns_qclass", "dns_qtype", "dns_rcode", "dns_AA", "dns_RD", "dns_RA", "dns_rejected"

a következő lépés a portok osztályozása, mivel a számértékük nem ad semmilyen tényleges információt az egymás közti kapcsolatról a modellnek
3 bucketbe rendezem őket:
0–1023      system/well-known
1024–49151  registered
49152+      ephemeral

### TODO

később a dst_portokat talán jobb lenne több kategóriába tenni?

## normalization

mivel a normalizálandó oszlopoknál
duration
src_bytes
dst_bytes
missed_bytes
src_pkts
src_ip_bytes
dst_pkts
dst_ip_bytes

nagyon nagy eltérések is vannak, ezért a minmax nem túl optimális választás, inkább log1p + standardscalert fogok alkalmazni