function [msg] = funname()
    db = dbstack('-completenames');
    assert(size(db,1)>=2);
    
    %packageName = db(2).
    pkg = fileparts(db(2).file);
    pkg = pkg(numel(cd)+2:end);
    pkg = strrep(pkg, '+', '');
    pkg = strrep(pkg, '\', '.');
    pkg = strrep(pkg, '/', '.');
    if(numel(pkg)>0)
         pkg = [pkg '.'];
    end
    
    clsFun = db(2).name;
    
    msg = [ pkg clsFun '()'];
    %msg = strrep(msg, '.', '::');% C++ style
end
