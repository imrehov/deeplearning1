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

droppoltam az oszlopokat
most a nem szám oszlopokat one-hot encodeolni kellene, ezek: "proto", "conn_state", "service", "dns_qclass", "dns_qtype", "dns_rcode", "dns_AA", "dns_RD", "dns_RA", "dns_rejected"

