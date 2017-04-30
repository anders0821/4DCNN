classdef ConvValidParallelGabor3 < handle
    properties
        pI;
        P;
        O;
        gradP;
        gradO;
    end
    
    methods
        function [this] = ConvValidParallelGabor3(pI, KERNELWIDTHS, pitchyaws, waveLengths, shifts)
            % 检查参数
            assert(all(size(KERNELWIDTHS)==[1 4]));
            assert(all(mod(KERNELWIDTHS, 2)==1));
            assert(KERNELWIDTHS(1)<=size(pI.O,1));
            assert(KERNELWIDTHS(2)<=size(pI.O,2));
            assert(KERNELWIDTHS(3)<=size(pI.O,3));
            assert(KERNELWIDTHS(4)==1);
            OUTPUTCHANNEL = size(pitchyaws, 1) * size(waveLengths, 2) * size(shifts, 2);
            
            % 初始化成员变量
            this.pI = pI;
            [H, W, D, T, INPUTCHANNEL, N] = size(this.pI.O);
            this.P = randn([KERNELWIDTHS INPUTCHANNEL OUTPUTCHANNEL]) * 0.01;
            this.O = zeros([H-KERNELWIDTHS(1)+1 W-KERNELWIDTHS(2)+1 D-KERNELWIDTHS(3)+1 T-KERNELWIDTHS(4)+1 OUTPUTCHANNEL N]);
            this.gradP = zeros(size(this.P));
            this.gradO = zeros(size(this.O));
            
            % 初始化固定核
            idx = 0;
            for shift=shifts
                for waveLength=waveLengths
                    for i=1:size(pitchyaws, 1)
                        pitch = pitchyaws(i, 1);
                        yaw = pitchyaws(i, 2);
                        R = [(KERNELWIDTHS(1)-1)/2 (KERNELWIDTHS(2)-1)/2 (KERNELWIDTHS(3)-1)/2];
                        K = util.gabor3_fwb([1 1], [yaw pitch], waveLength, shift, waveLength, R);% sigma==waveLength
                        idx = idx+1;
                        this.P(:,:,:,:,:,idx) = repmat(K, [1 1 1 1 INPUTCHANNEL]);
                    end
                end
            end
            assert(idx==OUTPUTCHANNEL);
            
            % 可视化核 z=0切片
            % for i=1:OUTPUTCHANNEL
            %     subplot(8,10,i);
            %     imshow(this.P(:,:,(1+end)/2,1,1,i), [-1 1]);
            % end
            % asd
        end
        
        function [] = ff(this)
            this.pI.ff();
            
            %fprintf('%s\t', util.funname());
            %tic;
            [H, W, D, T, INPUTCHANNEL, N] = size(this.pI.O);
            OUTPUTCHANNEL = size(this.P, 6);
            
            thispIO = this.pI.O;
            thisP = this.P;
            thisO = zeros([H-size(this.P,1)+1 W-size(this.P,2)+1 D-size(this.P,3)+1 T-size(this.P,4)+1 OUTPUTCHANNEL N]);
            parfor i=1:N
                for j=1:OUTPUTCHANNEL
                    for k=1:INPUTCHANNEL
                        tmp = convn(thispIO(:,:,:,:,k,i), thisP(:,:,:,:,k,j), 'valid');
                        thisO(:,:,:,:,j,i) = thisO(:,:,:,:,j,i) + tmp;
                    end
                end
            end
            this.O = thisO;
            
            %toc;
        end
        
        function [] = bp(this)
            warning(util.funname());
        end
        
        function [total, totalP] = debugMemory(this)
            [total, totalP] = this.pI.debugMemory();
            
            disp(util.funname());
            
            fprintf('	O');
            fprintf('	%d', size(this.O));
            fprintf('	(%d)', numel(this.O));
            fprintf('\n');
            
            fprintf('	P');
            fprintf('	%d', size(this.P));
            fprintf('	(%d)', numel(this.P));
            fprintf('\n');
            
            fprintf('	gradO');
            fprintf('	%d', size(this.gradO));
            fprintf('	(%d)', numel(this.gradO));
            fprintf('\n');
            
            fprintf('	gradP');
            fprintf('	%d', size(this.gradP));
            fprintf('	(%d)', numel(this.gradP));
            fprintf('\n');
            
            total = total + numel(this.O)+numel(this.gradO) + numel(this.P)+numel(this.gradP);
            totalP = totalP + numel(this.gradP);
        end
        
        function [p] = debugHist(this, fig1, fig2, fig3, fig4, m, n, p)
            p = this.pI.debugHist(fig1, fig2, fig3, fig4, m, n, p);
            
            p = p+1;
            
            if(~isempty(fig1))
                set(0, 'CurrentFigure', fig1);
                subplot(m,n,p);
                hist(this.O(:),100);
                title(class(this));
            end
            
            if(~isempty(fig2))
                set(0, 'CurrentFigure', fig2);
                subplot(m,n,p);
                hist(this.P(:),100);
                title(class(this));
            end
            
            if(~isempty(fig3))
                set(0, 'CurrentFigure', fig3);
                subplot(m,n,p);
                hist(this.gradO(:),100);
                title(class(this));
            end
            
            if(~isempty(fig4))
                set(0, 'CurrentFigure', fig4);
                subplot(m,n,p);
                hist(this.gradP(:),100);
                title(class(this));
            end
        end
        
        function [Ps] = getPs(this)
            [Ps] = this.pI.getPs();
        end
        
        function [] = setPs(this, Ps)
            this.pI.setPs(Ps);
        end
    end
end
