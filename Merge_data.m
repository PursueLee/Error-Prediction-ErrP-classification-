%%%%%%%%%% Merge raw EEG and Event %%%%%%%%%%%%%%

global sub2_1_rawEEG
global sub2_1_eve


%%%%%%%%%% Subject2 run1 %%%%%%%%%%%%%
sub2_1_1_rawEEG=run{1,1}.eeg;
sub2_1_1_eve=[run{1,1}.header.EVENT.POS run{1,1}.header.EVENT.TYP];

%%%%%%%%%% Subject2 run2 %%%%%%%%%%%%%
sub2_1_2_rawEEG=run{1,2}.eeg;
sub2_1_2_eve=[run{1,2}.header.EVENT.POS run{1,2}.header.EVENT.TYP];
[sub2_1_rawEEG,sub2_1_eve]=merge_arr(sub2_1_1_rawEEG,sub2_1_2_rawEEG, ...
                                     sub2_1_1_eve,sub2_1_2_eve);

%%%%%%%%%% Subject2 run3 %%%%%%%%%%%%%
sub2_1_3_rawEEG=run{1,3}.eeg;
sub2_1_3_eve=[run{1,3}.header.EVENT.POS run{1,3}.header.EVENT.TYP];
[sub2_1_rawEEG,sub2_1_eve]=merge_arr(sub2_1_rawEEG,sub2_1_3_rawEEG, ...
                                     sub2_1_eve,sub2_1_3_eve);

%%%%%%%%%% Subject2 run4 %%%%%%%%%%%%%
sub2_1_4_rawEEG=run{1,4}.eeg;
sub2_1_4_eve=[run{1,4}.header.EVENT.POS run{1,4}.header.EVENT.TYP];
[sub2_1_rawEEG,sub2_1_eve]=merge_arr(sub2_1_rawEEG,sub2_1_4_rawEEG, ...
                                     sub2_1_eve,sub2_1_4_eve);
                                 
%%%%%%%%%% Subject2 run5 %%%%%%%%%%%%%
sub2_1_5_rawEEG=run{1,5}.eeg;
sub2_1_5_eve=[run{1,5}.header.EVENT.POS run{1,5}.header.EVENT.TYP];
[sub2_1_rawEEG,sub2_1_eve]=merge_arr(sub2_1_rawEEG,sub2_1_5_rawEEG, ...
                                     sub2_1_eve,sub2_1_5_eve);
                                 
%%%%%%%%%% Subject2 run6 %%%%%%%%%%%%%
sub2_1_6_rawEEG=run{1,6}.eeg;
sub2_1_6_eve=[run{1,6}.header.EVENT.POS run{1,6}.header.EVENT.TYP];
[sub2_1_rawEEG,sub2_1_eve]=merge_arr(sub2_1_rawEEG,sub2_1_6_rawEEG, ...
                                     sub2_1_eve,sub2_1_6_eve);
                                 
%%%%%%%%%% Subject2 run7 %%%%%%%%%%%%%
sub2_1_7_rawEEG=run{1,7}.eeg;
sub2_1_7_eve=[run{1,7}.header.EVENT.POS run{1,7}.header.EVENT.TYP];
[sub2_1_rawEEG,sub2_1_eve]=merge_arr(sub2_1_rawEEG,sub2_1_7_rawEEG, ...
                                     sub2_1_eve,sub2_1_7_eve);
                                 
%%%%%%%%%% Subject2 run8 %%%%%%%%%%%%%
sub2_1_8_rawEEG=run{1,8}.eeg;
sub2_1_8_eve=[run{1,8}.header.EVENT.POS run{1,8}.header.EVENT.TYP];
[sub2_1_rawEEG,sub2_1_eve]=merge_arr(sub2_1_rawEEG,sub2_1_8_rawEEG, ...
                                     sub2_1_eve,sub2_1_8_eve);
                                 
%%%%%%%%%% Subject2 run9 %%%%%%%%%%%%%
sub2_1_9_rawEEG=run{1,9}.eeg;
sub2_1_9_eve=[run{1,9}.header.EVENT.POS run{1,9}.header.EVENT.TYP];
[sub2_1_rawEEG,sub2_1_eve]=merge_arr(sub2_1_rawEEG,sub2_1_9_rawEEG, ...
                                     sub2_1_eve,sub2_1_9_eve);
                                 
%%%%%%%%%% Subject2 run10 %%%%%%%%%%%%%
sub2_1_10_rawEEG=run{1,10}.eeg;
sub2_1_10_eve=[run{1,10}.header.EVENT.POS run{1,10}.header.EVENT.TYP];
[sub2_1_rawEEG,sub2_1_eve]=merge_arr(sub2_1_rawEEG,sub2_1_10_rawEEG, ...
                                     sub2_1_eve,sub2_1_10_eve);

%%%%%%%%%%%%% Convert the events(2D) into3D %%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%% Insert a 0 column between positon and type %%%%%%%%%%%%%%
%%%%%%%%%%%%% Here I operated this procedure in the command line %%%%%%

%{
sub2_1_eve(:,2+1)=0;
sub2_1_eve(:,[2,3])=sub1_1_eve(:,[3,2]);
%}

%%%%%%%%%%%%% Merge the total 10 runs into 1 structure%%%%%%%%%%%%%%%%

%{
Hence the position of each event is in samples and is corresponding to 
the start time of each event, when the events were merged, the events'
position should be modified.More specifically, add the total sampling
points of the last event.
%}

function[merged_EEG,merged_Eve]=merge_arr(arr1,arr2,arr3,arr4)
merged_EEG=[arr1;arr2];
persistent c;
c=zeros(size(arr4,1),size(arr4,2));
c(:,1)=size(arr1,1);
arr4=arr4+c;
merged_Eve=[arr3;arr4];
end






