function src_loc_start()
%SRC_LOC_START Initialize the toolbox
%   Details:
%       Initialization script for the src-localization-graphs toolbox. This
%       script adds to the path the folders with the files needed to run 
%       the toolbox. 
%
% Author: Rodrigo Pena (rodrigo.pena@epfl.ch)
% Date: 18 March 2016

%% Add dependencies
global GLOBAL_srcpath;
GLOBAL_srcpath = fileparts(mfilename('fullpath'));
addpath([GLOBAL_srcpath, ':', ...
    GLOBAL_srcpath, '/3rd_party:', ...
    GLOBAL_srcpath, '/demos:', ...
    GLOBAL_srcpath, '/experiments:', ...
    GLOBAL_srcpath, '/solvers:', ...
    GLOBAL_srcpath, '/utils:']);

end