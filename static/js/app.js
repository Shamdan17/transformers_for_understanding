var data = []
var token = ""

jQuery(document).ready(function () {
    var slider = $('#max_words')
    slider.on('change mousemove', function (evt) {
        $('#label_max_words').text('Top k choices: ' + slider.val())
    })

    var slider_mask = $('#max_words_mask')
    slider_mask.on('change mousemove', function (evt) {
        $('#label_max_words').text('Top k choices: ' + slider_mask.val())
    })

    $('#ca_input_text').on('keyup', function (e) {
        if (e.key == ' ') {
            $.ajax({
                url: '/get_end_predictions',
                type: "post",
                contentType: "application/json",
                dataType: "json",
                data: JSON.stringify({
                    "input_text": $('#ca_input_text').val(),
                    "question": $('#ca_input_q').val(),
                    "choices": $('#ca_input_choices').val(),
                    "top_k": slider.val(),
                }),
                beforeSend: function () {
                    $('.overlay').show()
                },
                complete: function () {
                    $('.overlay').hide()
                }
            }).done(function (jsondata, textStatus, jqXHR) {
                console.log(jsondata)
                $('#text_uqa_large').val(jsondata['uqalarge'])
                $('#text_uqa_3B').val(jsondata['uqa3b'])
                $('#text_T5_large').val(jsondata['t5_large'])
                $('#text_T5_3B').val(jsondata['t5_3b'])
                // $('#text_bert').val(jsondata['bert'])
                // $('#text_xlnet').val(jsondata['xlnet'])
                $('#text_xlm').val(jsondata['xlm'])
                // $('#text_bart').val(jsondata['bart'])
                // $('#text_electra').val(jsondata['electra'])
                $('#text_roberta').val(jsondata['roberta'])
            }).fail(function (jsondata, textStatus, jqXHR) {
                console.log(jsondata)
            });
        }
    })

        $('#ca_input_choices').on('keyup', function (e) {
        if (e.key == ' ') {
            $.ajax({
                url: '/get_end_predictions',
                type: "post",
                contentType: "application/json",
                dataType: "json",
                data: JSON.stringify({
                    "input_text": $('#ca_input_text').val(),
                    "question": $('#ca_input_q').val(),
                    "choices": $('#ca_input_choices').val(),
                    "top_k": slider.val(),
                }),
                beforeSend: function () {
                    $('.overlay').show()
                },
                complete: function () {
                    $('.overlay').hide()
                }
            }).done(function (jsondata, textStatus, jqXHR) {
                console.log(jsondata)
                $('#text_uqa_large').val(jsondata['uqalarge'])
                $('#text_uqa_3B').val(jsondata['uqa3b'])
                $('#text_T5_large').val(jsondata['t5_large'])
                $('#text_T5_3B').val(jsondata['t5_3b'])
                // $('#text_bert').val(jsondata['bert'])
                // $('#text_xlnet').val(jsondata['xlnet'])
                $('#text_xlm').val(jsondata['xlm'])
                // $('#text_bart').val(jsondata['bart'])
                // $('#text_electra').val(jsondata['electra'])
                $('#text_roberta').val(jsondata['roberta'])
            }).fail(function (jsondata, textStatus, jqXHR) {
                console.log(jsondata)
            });
        }
    })

        $('#ca_input_q').on('keyup', function (e) {
        if (e.key == ' ') {
            $.ajax({
                url: '/get_end_predictions',
                type: "post",
                contentType: "application/json",
                dataType: "json",
                data: JSON.stringify({
                    "input_text": $('#ca_input_text').val(),
                    "question": $('#ca_input_q').val(),
                    "choices": $('#ca_input_choices').val(),
                    "top_k": slider.val(),
                }),
                beforeSend: function () {
                    $('.overlay').show()
                },
                complete: function () {
                    $('.overlay').hide()
                }
            }).done(function (jsondata, textStatus, jqXHR) {
                console.log(jsondata)
                $('#text_uqa_large').val(jsondata['uqalarge'])
                $('#text_uqa_3B').val(jsondata['uqa3b'])
                $('#text_T5_large').val(jsondata['t5_large'])
                $('#text_T5_3B').val(jsondata['t5_3b'])
                // $('#text_bert').val(jsondata['bert'])
                // $('#text_xlnet').val(jsondata['xlnet'])
                $('#text_xlm').val(jsondata['xlm'])
                // $('#text_bart').val(jsondata['bart'])
                // $('#text_electra').val(jsondata['electra'])
                $('#text_roberta').val(jsondata['roberta'])
            }).fail(function (jsondata, textStatus, jqXHR) {
                console.log(jsondata)
            });
        }
    })



    // $('#btn-process').on('click', function () {
    //     $.ajax({
    //         url: '/get_mask_predictions',
    //         type: "post",
    //         contentType: "application/json",
    //         dataType: "json",
    //         data: JSON.stringify({
    //             "input_text": $('#mask_input_text').val(),
    //             "top_k": slider_mask.val(),
    //         }),
    //         beforeSend: function () {
    //             $('.overlay').show()
    //         },
    //         complete: function () {
    //             $('.overlay').hide()
    //         }
    //     }).done(function (jsondata, textStatus, jqXHR) {
    //         console.log(jsondata)
    //         $('#mask_text_bert').val(jsondata['bert'])
    //         $('#mask_text_xlnet').val(jsondata['xlnet'])
    //         $('#mask_text_xlm').val(jsondata['xlm'])
    //         $('#mask_text_bart').val(jsondata['bart'])
    //         $('#mask_text_electra').val(jsondata['electra'])
    //         $('#mask_text_roberta').val(jsondata['roberta'])
    //     }).fail(function (jsondata, textStatus, jqXHR) {
    //         console.log(jsondata)
    //     });
    // })
})