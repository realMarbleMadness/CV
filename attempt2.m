close all;
clear;
clc;

% load everything
bb1 = imread('realb1.png');
bb2 = imread('realb2.png');
bb3 = imread('realb3.png');


bb1 = im2double(bb1);
bw = im2bw(bb1, 0.42);
% operation on white
bw = imcomplement(bw);

figure, imshow(bw)

areas = regionprops(bw, 'Area');
avg_area = mean([areas(:).Area])

bw = bwareaopen(bw, round(avg_area/10));

figure, imshow(bw)

% hsv filter
all_hsv = rgb2hsv(bb1);
all_h = all_hsv(:,:,1)>0.11;
all_h = all_h.*1;

all_s = all_hsv(:,:,2)>0.2;  % baxter white board is dirty as fuck, need additional filtering
all_s = all_s.*1;

% by adding another all_s, we are essentially giving it more weight
mask = (bw + all_h + all_s + all_s) > 2; 

areas = regionprops(mask, 'Area');
avg_area = mean([areas(:).Area])
mask = bwareaopen(mask, round(avg_area/10));

% filtering will create holes, flood fill them
mask = imfill(mask,'holes');

se = strel('disk', 1);
mask = imerode(mask, se);
se = strel('disk', 10);
mask = imclose(mask, se);

areas = regionprops(mask, 'Area');
areas = [areas(:).Area]
orientations = regionprops(mask, 'Orientation');
orientations = [orientations(:).Orientation]
perimeters = regionprops(mask, 'Perimeter');
perimeters = [perimeters(:).Perimeter]
solidity = regionprops(mask, 'Solidity');
solidity = [solidity(:).Solidity]
AoP = areas./perimeters

mask = mask.*1;

all1 = bb1(:,:,1).*mask;
all2 = bb1(:,:,2).*mask;
all3 = bb1(:,:,3).*mask;
all_new = cat(3, all1, all2, all3);

% figure, imshow(all_new)

[numRow, numCol] = size(mask);
CC = bwconncomp (mask);
cont = zeros(numRow, numCol);
corners = zeros(CC.NumObjects, 4);                         
figure, imshow (all_new);
hold on;

positions = zeros(CC.NumObjects, 2);
messages = cell(CC.NumObjects,1);

for numChar = 1:CC.NumObjects  % cycle through all images

    curr = CC.PixelIdxList{numChar};
    [I, J] = ind2sub([numRow, numCol], curr);

    % add 5 pixel of padding
    corners(numChar,:) = [min(J)-10, min(I)-10, max(J)+10, max(I)+10]; 

    cont(I, J) = 1;

    rectangle('Position', [corners(numChar,1) corners(numChar,2) corners(numChar,3)-corners(numChar,1) corners(numChar,4)-corners(numChar,2)], ...
          'EdgeColor','r', 'LineWidth',3);

    positions(numChar,:) = [min(J)-10, max(I)-10];
    
    % classify
    if (solidity(numChar) > 0.9)% && AoP(numChar) > 19)
        messages{numChar} = ['long rectangle'];
    elseif (solidity(numChar) < 0.65) %% && AoP(numChar) < 13)
        messages{numChar} = ['goal'];
    elseif (solidity(numChar) > 0.75 && solidity(numChar) < 0.85) % && AoP(numChar) > 14 && AoP(numChar) < 17)
        messages{numChar} = ['small arc'];
    elseif (solidity(numChar) < 0.73) %% && AoP(numChar) > 17 && AoP(numChar) < 20)
        messages{numChar} = ['big arc'];
    else
        messages{numChar} = ['wtf is this'];
    end
    
    hold on;

end

RGB = insertText(all, positions, messages,'AnchorPoint','LeftBottom');
figure, imshow(RGB);






