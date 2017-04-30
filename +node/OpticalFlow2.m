classdef OpticalFlow2 < handle
    properties
        shape;
        
        pI;
        O;
        gradO;
    end
    
    methods
        function [this] = OpticalFlow2(pI, shape)
            this.pI = pI;
            this.shape = shape;
            assert(strcmp(shape,'valid') || strcmp(shape,'same'));
            
            [H, W, D, T, C, N] = size(this.pI.O);
            assert(T==1);
            assert(C==1)
            
            if(strcmp(this.shape,'same'))
                % Dͨ���ϼ������ ����֡����Ϊ�ڶ�֡���� ��֤Dά�Ȳ���
                % �����ͨ��Ϊ��������Ĺ���1->2
                this.O = zeros([H W D T 2 N]);
                this.gradO = zeros(size(this.O));
            else
                this.O = zeros([H W D-1 T 2 N]);
                this.gradO = zeros(size(this.O));
            end
        end
        
        function [] = ff(this)
            this.pI.ff();
            
            %fprintf('%s\t', util.funname());
            %tic;
            
            [H, W, D, T, C, N] = size(this.pI.O);
            assert(T==1);
            assert(C==1);
            of = zeros(H, W, D-1, T, 2, N);% of��������һ֡ ��2ͨ��
            for i=1:N
                % ������������
                optical_flow_calculator = vision.OpticalFlow('Method','Horn-Schunck',... % HS����
                    'ReferenceFrameSource', 'Property', 'ReferenceFrameDelay', 1,... % ��֡������� ʹ�����û���
                    'Smoothness', 1,...
                    'IterationTerminationCondition', 'Maximum iteration count', 'MaximumIterationCount', 10, ... % ������ֹ����
                    'OutputValue',  'Horizontal and vertical components in complex form' ... % ���������������
                    );
                
                % ��֡�����
                for j=1:D
                    im = this.pI.O(:,:,j,:,:,i);
                    optical_flow_vector = step(optical_flow_calculator, im);
                    if(j>=2)
                        of(:,:,j-1,:,1,i) = real(optical_flow_vector);
                        of(:,:,j-1,:,2,i) = imag(optical_flow_vector);
                    end
                end
            end
            
            if(strcmp(this.shape,'same'))
                % ����֡����Ϊ�ڶ�֡���� ��֤Dά�Ȳ���
                this.O(:,:,1,:,:,:) = of(:,:,1,:,:,:);
                this.O(:,:,2:end,:,:,:) = of;
            else
                assert(all(all(all(all(all(all(size(this.O)==size(of))))))));
                this.O = of;
            end
            
            % % ���ӻ�this.O
            % OUTPUTCHANNEL = size(this.O, 5);
            % for i=1:N
            %     figure;
            %     for j=1:OUTPUTCHANNEL
            %         for k=1:D
            %             subplot(OUTPUTCHANNEL,D,(j-1)*D+k);
            %             im = this.O(:,:,k,1,j,i);
            %             if(j==3)
            %                 imshow(im);
            %             else
            %                 imshow(im, [-0.1 0.1]);
            %             end
            %         end
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
