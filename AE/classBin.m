function [binned, classes] = classBin(values,classLabels)
    classes = unique(classLabels);
    nClasses = length(classes);
    
    binned = cell(1, length(classes));
    
    for i = 1:nClasses
        binned{i} = values(classLabels==classes(i));
    end
end

