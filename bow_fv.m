clear
addpath(genpath('./fun'))
datasets={'shoes','clothes'};
for k=1:2
dataset=datasets{k};
load('../meta-data/Boxes_for_eachsearch.mat');
load('../meta-data/img_names.mat');
load(['../meta-data/' dataset 'vocab.mat']);
%{
descrs=cell(10010,1);
c=1;
for i=1:10
    img=imcrop(imread(['../search' dataset '/' name.(['search' dataset])(i).name]),...
        name.(['search' dataset])(i).BoundingBox);
    im=im2single(img);
    [~, descrs{c}] = vl_phow(im, 'Sizes',[3 5 7],'step',3);
    c=c+1;
end
for i=1:10
    box=Boxes.(dataset){i};
    idx=randperm(15000,1000);
    box=box(idx,:);
    for j=1:1000
        img=imcrop(imread(['../' dataset '/' name.(dataset)(box(j,5)).name]),...
            box(j,1:4));
        im=im2single(img);
        [~, descrs{c}] = vl_phow(im, 'Sizes',[3 5 7],'step',3);
        c=c+1;
    end
end
ftrs=cat(2,descrs{:});
clear descrs
ftr=vl_colsubset(ftrs, 3000000);
clear ftrs
[means, covariances, priors] = vl_gmm(single(ftr), 64);
save(['../meta-data/' dataset 'vocab.mat'],'means','covariances','priors')
%}
ftr=zeros(10,16384,'single');
c=1;
for i=1:10
    img=imcrop(imread(['../search' dataset '/' name.(['search' dataset])(i).name]),...
        name.(['search' dataset])(i).BoundingBox);
    im=im2single(img);
    [~, desc] = vl_phow(im, 'Sizes',[3 5 7],'step',3);
    encoding = vl_fisher(single(desc), means, covariances, priors,'Improved');
     ftr(c,:)=encoding';
     c=c+1;
end
Bow.(['search'  dataset])=ftr;
Bow.(dataset)=zeros(150000,16384,'single');
c=1;
for i=1:10
    box=Boxes.(dataset){i};
    for j=1:15000
        disp([ dataset ':' num2str(i) ' - ' num2str(j)])
        img=imcrop(imread(['../' dataset '/' name.(dataset)(box(j,5)).name]),...
            box(j,1:4));
        im=im2single(img);
        [~, desc] = vl_phow(im, 'Sizes',[3 5 7],'step',3);
        encoding = vl_fisher(single(desc), means, covariances, priors,'Improved');
        Bow.(dataset)(c,:)=encoding';
        c=c+1;
    end
end
end
save(['../meta-data/Bow_phow_fv.mat'],'-struct','Bow','-v7.3')
