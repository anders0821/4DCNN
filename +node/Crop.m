classdef Crop < handle
    properties
        shape;
        
        pI;
        O;
        gradO;
    end
    
    methods
        function [this] = Crop(pI, SPATIAL_OUTPUT_SIZE, shape)
            this.pI = pI;
            this.shape = shape;
            assert(strcmp(shape,'center') || strcmp(shape,'rand'));
            
            [H, W, D, T, C, N] = size(this.pI.O);
            assert(all(SPATIAL_OUTPUT_SIZE<=[H W D T]));
            
            this.O = zeros([SPATIAL_OUTPUT_SIZE C N]);
            this.gradO = zeros(size(this.O));
        end
        
        function [] = ff(this)
            this.pI.ff();
            
            %fprintf('%s\t', util.funname());
            %tic;
            
            [H, W, D, T, C, N] = size(this.pI.O);
            [H2, W2, D2, T2, C, N] = size(this.O);
            
            for i=1:N
                sMax = [H W D T]-[H2 W2 D2 T2]+[1 1 1 1];
                sMin = [1 1 1 1];
                if(strcmp(this.shape,'rand'))
                    s = [randi([sMin(1) sMax(1)]) randi([sMin(2) sMax(2)]) randi([sMin(3) sMax(3)]) randi([sMin(4) sMax(4)])];
                else
                    s = (sMin+sMax)/2;
                end
                e = s + [H2 W2 D2 T2] - 1;
                this.O(:,:,:,:,:,i) = this.pI.O(s(1):e(1), s(2):e(2), s(3):e(3), s(4):e(4), :, i);
            end
            
            % size(this.pI.O)
            % size(this.O)
            % for i=1:size(this.O, 5)
            %     for j=1:size(this.O, 3)
            %         subplot(size(this.O, 5), size(this.O, 3), (i-1)*size(this.O, 3)+j);
            %         imshow(this.O(:,:,j,:,i,1)*10);
            %     end
            % end
            % asd
            
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
            
            fprintf('	gradO');
            fprintf('	%d', size(this.gradO));
            fprintf('	(%d)', numel(this.gradO));
            fprintf('\n');
            
            total = total + numel(this.O)+numel(this.gradO);
            totalP = totalP + 0;
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
            
            if(~isempty(fig3))
                set(0, 'CurrentFigure', fig3);
                subplot(m,n,p);
                hist(this.gradO(:),100);
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
