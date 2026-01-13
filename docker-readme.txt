# build docker image
docker build . -f Dockerfile -t kpimpc:2026.1.13.dev

# test docker image
docker run -i -t --rm -v /home/datawork-cersat-public/:/home/datawork-cersat-public/ kpimpc:2026.1.13.dev /bin/bash
or
docker run -i -t --rm -v /home/datawork-cersat-public/:/home/datawork-cersat-public/ kpimpc:2026.1.13.dev kpihs -h
example 
docker run -i -t --rm -v /raid:/raid -v /home/datawork-cersat-public/:/home/datawork-cersat-public/ kpimpc:2026.1.13.dev kpihs  --overwrite --satellite S1A --wv wv1 --enddate 20221214 --outputdir /raid/localscratch/agrouaze/

