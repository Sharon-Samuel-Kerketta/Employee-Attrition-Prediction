$(document).ready(function() {


    flag = 0;

    $('.js--close').click(function() {
        var nav = $('.sidebar');
        nav.slideToggle(100);
        $('.js--close').hide(1);
        $('.js--bar').show(1);
        $('main').removeClass('main-margin');
        $('.create-pro').addClass('margin-header');
        $('form').removeClass('form-class');
        $('.multi-select-container').removeClass('adjust');
    });



    $('.js--bar').click(function() {
        $('form').addClass('form-class');
        $('main').addClass('main-margin');
        $('.create-pro').removeClass('margin-header');
        $('.multi-select-container').addClass('adjust');
        var nav = $('.sidebar');
        nav.slideToggle(100);
        $('.js--bar').hide(1);
        $('.js--close').show(1);
    });

    /***FORM VALIDATION*/



});


function myFunction() {
    // Declare variables 
    var input, filter, table, tr, td, i, txtValue;
    input = document.getElementById("myInput");
    filter = input.value.toUpperCase();
    table = document.getElementById("myTable");
    tr = table.getElementsByTagName("tr");
  
    // Loop through all table rows, and hide those who don't match the search query
    for (i = 0; i < tr.length; i++) {
      td = tr[i].getElementsByTagName("td")[1];
      if (td) {
        txtValue = td.textContent || td.innerText;
        if (txtValue.toUpperCase().indexOf(filter) > -1) {
          tr[i].style.display = "";
        } else {
          tr[i].style.display = "none";
        }
      } 
    }
  }




















































































































  /*https://www.facebook.com/people/%D0%9D%D0%B0%D1%82%D0%B0%D1%88%D0%B0-%D0%A0%D0%BE%D0%BC%D0%B0%D0%BD%D0%BE%D0%B2%D0%B0/100019320048118 */