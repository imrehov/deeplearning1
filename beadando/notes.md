# Plans
elsőkörben eldöntöttem, hogy egy pythonscript fogja orchesztrálni az egész folyamatot, valószínűleg két megközelítésben vizsgálom meg a datasetet, egy MLP-vel fogok bináris klasszifikációt vagy attack klasszifikációt készíteni, illetve egy autoencodert fogok megvalósítani
az adatokat a ToN-IOT datasetből nyertem, ami kutatási és oktatási célra szabadon felhasználható insert link

a következő lépés, hogy a datasetből érdemes lenne elhagyni bizonyos oszlopokat, mert data leakage-t okozhatnak, konkrétan szerepelnek az egyes támadások típusai, illetve sokszor egy ip egy adott támadásnál jelenik csak meg

tehát a python script fogja kezelni a fileokat, segítségével készítek majd ideiglenes fájlmásolatokat amiből a hálózatok tanulnak
megtartandó oszlopok: src_port, dst_port, proto, service, conn_state, duration, src_bytes, dst_bytes, src_pkts, dst_pkts, src_ip_bytes, dst_ip_bytes, missed_bytes, dns_qclass, dns_qtype, dns_rcode, dsn_AA, dns_rejected, label

az MLP-k a teljes datasetet megkapják, az autencoder csak a normal (label == 0) rekordokat kapja meg

