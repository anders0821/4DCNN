classdef Data < handle
    properties
        fn;
        varName;
        dim;
        
        MINIBATCHSIZE;
        miniBatchOffset;
        O;
        gradO;
    end
    
    methods
        function [this] = Data(fn, varName, MINIBATCHSIZE)
            % 获取数据维度
            this.fn = fn;
            this.varName = varName;
            this.dim = util.hdf5getdim(fn, varName);
            assert(MINIBATCHSIZE>=1);
            assert(MINIBATCHSIZE<=this.dim(6));
            
            % 初始化成员变量
            this.MINIBATCHSIZE = MINIBATCHSIZE;
            this.miniBatchOffset = 1;
            this.O = zeros(this.dim(1), this.dim(2), this.dim(3), this.dim(4), this.dim(5), MINIBATCHSIZE);
            this.gradO = zeros(size(this.O));
        end
        
        function [] = ff(this)
            %fprintf('%s\t', util.funname());
            %tic;
            
            % 计算miniBatchIdx
            MAX = this.dim(6);
            S = this.miniBatchOffset;
            L = this.MINIBATCHSIZE;
            E = this.miniBatchOffset+L-1;
            if(E<=MAX)
                % [S L S+L-1]
                this.O = h5read(this.fn, ['/' this.varName], [1 1 1 1 1 S], [this.dim(1) this.dim(2) this.dim(3) this.dim(4) this.dim(5) L]);
            else
                L1 = MAX-S+1;
                L2 = L-L1;
                % [S L1 S+L1-1]
                % [1 L2 1+L2-1]
                part1 = h5read(this.fn, ['/' this.varName], [1 1 1 1 1 S], [this.dim(1) this.dim(2) this.dim(3) this.dim(4) this.dim(5) L1]);
                part2 = h5read(this.fn, ['/' this.varName], [1 1 1 1 1 1], [this.dim(1) this.dim(2) this.dim(3) this.dim(4) this.dim(5) L2]);
                this.O = cat(6, part1, part2);
            end
            this.O = double(this.O);
            assert(all( size(this.O)==[this.dim(1) this.dim(2) this.dim(3) this.dim(4) this.dim(5) this.MINIBATCHSIZE] ));
            assert(isa(this.O, 'double'));
            
            % 偏移miniBatchOffset
            this.miniBatchOffset = mod(E, MAX)+1;
            %toc;
        end
        
        function [] = bp(this)
            warning(util.funname());
        end
        
        function [total, totalP] = debugMemory(this)
            disp(util.funname());
            
            fprintf('	data');
            fprintf('	%d', this.dim);
            fprintf('	(%d)', prod(this.dim));
            fprintf('\n');
            
            fprintf('	O');
            fprintf('	%d', size(this.O));
            fprintf('	(%d)', numel(this.O));
            fprintf('\n');
            
            fprintf('	gradO');
            fprintf('	%d', size(this.gradO));
            fprintf('	(%d)', numel(this.gradO));
            fprintf('\n');
            
            total = numel(this.O)+numel(this.gradO);
            totalP = 0;
        end
        
        function [p] = debugHist(this, fig1, fig2, fig3, fig4, m, n, p)
            %p = this.pI.debugHist(fig1, fig2, fig3, fig4, m, n, p);
            
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
            Ps = cell(0);
        end
        
        function [] = setPs(this, Ps)
        end
    end
end
