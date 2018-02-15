/**
 * Makes an AJAX call and puts the result in any container
 * 
 * @param link the link to the webpage
 * @param target -optional- the container where to put the result
 * @param completionFunction -optional- a function that does postprocessing after the DOM is created
 * @param asynchronousCall -optional- if we want a blocking or non-blocking request
 */
function ajaxCall_generic(link, target, completionFunction, asynchronousCall) {
    var isAsync = true;
    if (asynchronousCall)
        isAsync = async;
    if (target)
        $(target).html("<img src='images/ajax-loader.gif'/>");
    $.ajax({
        type : 'POST',
        url : link,
        cache : false,
        async: isAsync,
        success : function(response) {
            if (target)
                $(target).html(response);
            if (completionFunction)
                completionFunction(response);
        },
        complete : function() {
            return true;
        }
    });
}
