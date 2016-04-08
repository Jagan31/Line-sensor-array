%JAGADEESHWARAN R and LOKESH KUMAR A N 
%DATE: 18:02:2016

%Face Recognition Using Particle Swarm Optimization-Based Selected Features

clc; clear all; close all;

path_name = 'D:\assignment 5\fert\f';

%%============Global and Other Parameters Initialization=================%%

global classmeanarray totalmeanarray GlobalBestP;
t1=zeros(1,10);
t3=zeros(1,10);
countsum=0;
percentsum=0;
f = fspecial('gaussian',[3 3],0.5);
b = zeros(35,8);
for i= 1:280
storeface{i} = zeros(1,96);
copy{i} = zeros(1,96);
end
%%=======================================================================%%
                       %Train and Test Iterations%
%%=======================================================================%%

for x = 1:1                                            %5 Iterations

    
%figure;                                               %for displaying the
%str=sprintf('failure images of Iteration No.%d ',x);  %failed cases of the
%suptitle(str)                                         %test images             

z=0;
tic;                                        %start testing simulation time
ttotal = zeros(12,8);
k = 1;

for j = 1:35                              %35 subjects

%======================Acquire test images Randomly=======================%    
    
    b(j,:) = randperm(20,8);
    tsum = zeros(12,8);
    for i = 1:8                           %Eight testing images per subject    

    face=imread(strcat(path_name,...
    num2str(j),'\',num2str(b(j,i)),'.ppm'));

%==========================Preprocessing steps============================%    
    
    %face = imadjust(face,[],[],(1/2.8)); %Gamma Intensity Correction
    %face = imfilter(face,f,'replicate'); %Gaussian Blurr
    
    f1 = flipdim(face ,2);                %avg of fliped and original image
    f2=(f1+face)/2;                       
    face=rgb2gray(f2);                    %convert to grayscale 
    face=imresize(face,[96 64]);          

%==========================Feature extraction=============================%    
%---------------------3-Lvl Wavelet Decompositiion------------------------%
   wname = 'coif5';
   [CA1,CH1,CV1,CD1] = dwt2(face,wname);
   [CA2,CH2,CV2,CD2] = dwt2(CA1,wname);
   [CA,CH,CV,CD] = dwt2(CA2,wname);
   
%-------------------------------------------------------------------------%  
   u=imresize(CA,[12 8]);
   storeface{k} = reshape(u,1,96);       %Store in 1x96 vector
   copy{k}=storeface{k};
   k = k+1;
   tsum = double(tsum)+double(u);        
   end
 ttotal = double(tsum)+double(ttotal);    
 avg = (tsum/8);
 classmeanarray{j} = avg;                %Mean of DCT coefficients for 
                                         %each Class(Subject) 
end

avgall = ttotal/280;                        
totalmeanarray = avgall;                 %Mean of DCT coefficints of all 
                                         %Classes
                                         
%%==============================Start BPSO===============================%%                                         

%------------------------Initalization of Parameters----------------------%

NPar = 96;                               %Number of Dimensional Parameters
NumofParticles = 30;                     %Number of Particles
Velocity = zeros(NumofParticles,NPar);
Position = zeros(NumofParticles,NPar);
Cost = zeros(NumofParticles,1);
LocalBestCost = zeros(NumofParticles,1);
LocalBestPosition = zeros(NumofParticles,NPar);
ff='fitness4';                            %Fitness function
GlobalBestP = rand(1,NPar);              
GlobalBestC = 0; 

MaxIterations = 30;                       %Number of BPSO iterations
Damp=0.88;                                %Inertial Damping Factor
C1 = 1.6182;                              %Cognitive Factor
C2 = 0.6192;                              %Social Factor

%-------------------------Initialization of Particles---------------------%

for i = 1:NumofParticles
    Velocity(i,:) = (rand(1,NPar));
    R = rand(1,NPar);
    Position(i,:) = R < 1./(1 + exp(-Velocity(i,:)));
    Cost(i,:) = feval(ff,Position(i,:),10,NPar);
    LocalBestPosition(i,:) = Position(i,:);
    LocalBestCost(i,:) = Cost(i,:);

    if Cost(i,:) > GlobalBestC
        GlobalBestP = Position(i,:);
        GlobalBestC = Cost(i,:);
    end
end
%----------------------------Start BPSO iterations------------------------%
for t = 1:MaxIterations
    Damp=Damp.^t;
    for i = 1:NumofParticles
        r1 = rand(1,NPar);
        r2 = rand(1,NPar);
        w = rand(1,NPar);
        Velocity(i,:) = Damp*Velocity(i,:) + ...
            r1*C1.*(LocalBestPosition(i,:) - Position(i,:)) + ...
            r2*C2.*(GlobalBestP - Position(i,:));
         
        R = rand(1,NPar);
        Position(i,:) = R < 1./(1 + exp(-Velocity(i,:)));
        Cost(i,:) =feval(ff,Position(i,:),10,NPar);
       
        
        if Cost(i,:) > LocalBestCost(i,:);
            LocalBestPosition(i,:) = Position(i,:);
            LocalBestCost(i,:) = Cost(i,:);
            if Cost(i,:) > GlobalBestC
                GlobalBestP = Position(i,:);
                GlobalBestC = Cost(i,:);
            end
        end   
    end

 end
%--------------------------------End BPSO---------------------------------%

%%==========================Results from BPSO============================%%

count = length(find(GlobalBestP));           %Number of Features selected 
disp('Number of selected features:');
disp(count);
temp(x)=count;

for t= 1:280
    storeface{t}= storeface{t}.*GlobalBestP;  %Optimized Feature vector 
end                                           %for each Face


t1(x)=toc;                                  %stop training simulation time

%%=============================Start Testing=============================%%


tic;                                         %start testing simulation time

rec=0;                                       %Recognition Counter
tests=420;                                   %Run test for remaining 420
                                             %images
for n=1:tests                       
    c = ceil(n/12); 
    b2 = 1:20;  
    b1 = setdiff(b2,b(c,:));                 %Select images not used in 
                                             %testing stages
                                             
    i = mod(n,12)+(12 * (mod(n,12)==0));        
    
    img = imread(strcat(path_name,...
           num2str(c),'\',num2str(b1(i)),'.ppm'));
       
       
       
%-----------------------------Preprocessing-------------------------------%    
    
    %img = imadjust(img,[],[],(1/2.8));
    %img = imfilter(img,f,'replicate');
    i1=flipdim(img,2);                    
    i2=(i1+img)/2;                          %avg of fliped and original image
    img=rgb2gray(i2);                       %convert to grayscale
    img=imresize(img,[96 64]);            
    
%--------------------------Feature Extraction-----------------------------%
 
%---------------------3-Lvl Wavelet Decompositiion------------------------%
wname = 'haar';                            %haar wavelet transform
 [CA11,CH11,CV11,CD11] = dwt2(img,wname);
 [CA22,CH22,CV22,CD22] = dwt2(CA11,wname);
 [CA3,CH3,CV3,CD3]     = dwt2(CA22,wname);
%-------------------------------------------------------------------------%
  q=imresize(CA3,[12 8]);
  pic_dct=reshape(q,1,96);              %convert to vector
  

%--------------------------Feature Selection------------------------------%    
    
    opt=pic_dct.*GlobalBestP;
    
%-------------Compute Euclidean Distance with each test vector------------%

     d=zeros(1,280);
 
          for p=1:280 
             r = storeface{p};
             d(p) = sqrt(sum((r-opt).^2));    
          end 
             
     [val,index]=min(d);                   %Minimum of Euclidean Distances
                                           %gives the Matched Vector
                                           
                                           
                                           
%=======================Display the failure cases(=========================%
     
       %if((ceil(index/4))~=c)
           
       % z=z+1;
       % subplot(5,5,z); 
       % img = imshow(strcat(path_name,...
       % num2str(c),'\',num2str(b1(i)),'.pgm'));
       % title(strcat('s',num2str(c),'-',num2str(b1(i)),'.pgm'))
        
       %end
%-------------------------------------------------------------------------%   
     

     if((ceil(index/8))==c)                %Increment Recognition
     rec=rec+1;                            %Counter if successful  
     end                                   %Recognition
                                
    
end 
  
%=======================End of One Train-Test iteration===================%
 t2=toc;                                    %stop testing simulation time
 t3(x)=t2;
 
disp('Recognition rate:');                 %Find Recognition Rate 
percent=(rec/tests)*100;
disp(percent);
percentsum(x)=percent;

end
%%=======================================================================%%
                        %End of 'x' Train-test iterations%
%%=======================================================================%%



disp('Average number of selected features:')   %Find average number of
disp(sum(temp)/max(x));                        %selected features

disp('Average Recognition Rate:')              %Find average of
disp(sum(percentsum)/max(x));                  %Recognition rate  

disp('Average training simulation time in sec:')  %Find average training 
disp(sum(t1)/max(x));                             %simulation time

disp('Average testing simulation time of one test image in sec:') 
disp(sum(t3)/(max(x)*240));                     %Find average testing
                                                %simulation time per image

%-------------------------------------------------------------------------%   
