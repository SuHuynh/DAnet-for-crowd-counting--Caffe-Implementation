clc; clear all;
seed = 95461354;
rng(seed)
N = 15;
dataset = 'B';
path = ['D:\Su\UAV project\data_set\ShanghaiTech\part_' dataset '\train_data\images\'];
gt_path = ['D:\Su\UAV project\data_set\ShanghaiTech\part_' dataset '\train_data\ground-truth\'];
perspective_path = ['D:\Su\UAV project\data_set\perspective-ShanghaiTech\B\train_pmap\IMG_'];

output_path = 'D:\Su\UAV project\data_train\SHB_same_kernel\Original\data\';

train_path_img = strcat(output_path,'train_images\');
train_path_den = strcat(output_path,'train_csv_den\');
train_path_per = strcat(output_path, 'train_perspective\');

mkdir(output_path);
mkdir(train_path_img);
mkdir(train_path_den);
mkdir(train_path_per);


if (dataset == 'A')
    num_images = 300;
else
    num_images = 400;
end

indices = randperm(num_images);

for idx = 1:num_images
    i = indices(idx);
    if (mod(idx,1)==0)
        fprintf(1,'Processing %3d/%d files\n', i, num_images);
    end
    
    %load gt      
    load(strcat(gt_path, 'GT_IMG_',num2str(i),'.mat')) ;
    %load image     
    input_img_name = strcat(path,'IMG_',num2str(i),'.jpg');
    im = imread(input_img_name);
    [h, w, c] = size(im);
               
    if (c == 3)
         im_gray = rgb2gray(im);      
    end
    
    %load perspective map gt
    perspective_gt_name = strcat(perspective_path, num2str(i), '.mat');
    p_gt_map = load(perspective_gt_name);
    p_gt_norm = 1 - (p_gt_map.pmap)/255;
    p_gt_norm(p_gt_norm==1)=0;
    
    wn2=256;
    hn2=192;

    annPoints =  image_info{1}.location;         

    if( w <= 2*wn2 )
        im = imresize(im,[ h,2*wn2+1]);
        annPoints(:,1) = annPoints(:,1)*2*wn2/w;
    end
    if( h <= 2*hn2)
        im = imresize(im,[2*hn2+1,w]);
        annPoints(:,2) = annPoints(:,2)*2*hn2/h;
    end
    [h, w, c] = size(im);
    a_w = wn2+1; b_w = w - wn2;
    a_h = hn2+1; b_h = h - hn2;
    
    im_density = get_density_map_gaussian(im_gray,annPoints);
    
%     if(mod(idx,1)==0)
%         figure(1);
%         imagesc(im_density);
%     end

    for j = 1:N
        
        r=0.28*randn + 0.3;
        if(r>1)
            r=1;
        elseif(r<0)
            r=0;
        end
                  
        x = floor((b_w - a_w) * rand + a_w);
        y = floor((b_h - a_h) * r + a_h);
        
        x1 = x - wn2; y1 = y - hn2;
        x2 = x + wn2-1; y2 = y + hn2-1;
        
        
        im_sampled = im(y1:y2, x1:x2,:);
        im_density_sampled = im_density(y1:y2,x1:x2);
        p_gt_map_sampled = p_gt_norm(y1:y2,x1:x2);
        

        
        annPoints_sampled = annPoints(annPoints(:,1)>x1 & ...
            annPoints(:,1) < x2 & ...
            annPoints(:,2) > y1 & ...
            annPoints(:,2) < y2,:);
        annPoints_sampled(:,1) = annPoints_sampled(:,1) - x1;
        annPoints_sampled(:,2) = annPoints_sampled(:,2) - y1;
          
        img_idx = strcat(num2str(i), '_',num2str(j));        

        imwrite(im_sampled, [train_path_img num2str(img_idx) '.jpg']);
        csvwrite([train_path_den num2str(img_idx) '.csv'], im_density_sampled);
        imwrite(p_gt_map_sampled, [train_path_per num2str(img_idx) '.png']);
    end

    
end
