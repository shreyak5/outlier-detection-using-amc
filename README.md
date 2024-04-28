# Outlier Detection using Saliency Detection via Absorbing Markov Chains
Course Project for CS5150 - Applications of Markov Chain in Computing

**Work by:** \
Shreya Kumar - ES20BTECH11026 \
Vikhyath Kothamasu - CS20BTECH11056 \
U Adithyan - CS23BTNCL11001 

## How to run the code

(i) Generating the Heatmaps

(ii) Generating Boundary Map: 
- Change the input and output path as required in `edges-master/edgesDemo.m` and run the file.
- Ensure that the boundary map is stored under `boundary-map/` with the same name as the input image.


(iii) Generating Saliency Map
- Store the input images (.png format) in `image/`
- Store their corresponding boundary maps in `boundary-map/`
- Store their FCN-features in `FCN-feature/6/` and `FCN-feature/32/`.
- Run `Saliency_AMC_AE.m`


(iv) Making Final Inferences

## References:
[1] B. Jiang, L. Zhang, H. Lu, C. Yang and M. -H. Yang, "Saliency Detection via Absorbing Markov Chain," 2013 IEEE International Conference on Computer Vision, Sydney, NSW, Australia, 2013, pp. 1665-1672, doi: 10.1109/ICCV.2013.209. 

[2] Zitnick, C.L., Dollár, P. (2014). Edge Boxes: Locating Object Proposals from Edges. In: Fleet, D., Pajdla, T., Schiele, B., Tuytelaars, T. (eds) Computer Vision – ECCV 2014. ECCV 2014. Lecture Notes in Computer Science, vol 8693. Springer, Cham. https://doi.org/10.1007/978-3-319-10602-1_26

**Note:**
1. The saliency maps were generated using [1]
2. Boundary maps required for [1] were generated using [2]
