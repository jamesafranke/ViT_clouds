# Self-supervised cloud semantic segmentation with vision transformers

vision transformers trained without explicit supervision based on the DINO framework from 
https://arxiv.org/abs/2104.14294

applied to MODIS satelite images of derived cloud properties: 
*https://ladsweb.modaps.eosdis.nasa.gov/missions-and-measurements/products/MOD06_L2
*https://ladsweb.modaps.eosdis.nasa.gov/missions-and-measurements/products/MYD06_L2

and to level 1b radiances: 


# workflow

1. Download the raw MODIS from NASA with login

2. Reproject to uniform lat-lon grid

3. engineer training stacks, normalize
    a. liquid water path, ice water path, cloud top pressure
    b. RGB
    c. some other bands? 


open questions: 

* how many heads in last layer (implicit number of classes)?

* how to scale to 5kx5k pixel images?

* 
....
