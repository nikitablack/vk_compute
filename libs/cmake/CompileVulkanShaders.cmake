find_package(Vulkan REQUIRED)

if (NOT Vulkan_glslangValidator_FOUND)
    message(FATAL_ERROR "Failed to find glslangValidator.")
endif ()

function(addCompileShadersCommand)
    cmake_parse_arguments(addCompileShadersCommand "" "TARGET_NAME;OUTPUT_DIR" "SHADERS;DEPENDS" ${ARGN})

    if (NOT addCompileShadersCommand_TARGET_NAME)
        message(FATAL_ERROR "Provide unique target name.")
    endif ()

    set(custom_target "${addCompileShadersCommand_TARGET_NAME}_custom_target")

    if (NOT addCompileShadersCommand_SHADERS)
        message(FATAL_ERROR "At least one shader file name should be provided.")
    endif ()

    set(FILE_NAMES "")
    set(OUTPUT_PATHS "")

    foreach (shader ${addCompileShadersCommand_SHADERS})
        if(NOT EXISTS ${shader})
            message(FATAL_ERROR "Failed to find shader file ${shader}.")
        endif()

        file(MAKE_DIRECTORY ${addCompileShadersCommand_OUTPUT_DIR})

        get_filename_component(VAR ${shader} NAME)
        list(APPEND FILE_NAMES ${VAR})
        list(APPEND OUTPUT_PATHS ${addCompileShadersCommand_OUTPUT_DIR}/${VAR}.spv)

        add_custom_command(OUTPUT ${addCompileShadersCommand_OUTPUT_DIR}/${VAR}.spv
                COMMAND ${Vulkan_GLSLANG_VALIDATOR_EXECUTABLE} -gVS --target-env vulkan1.3 -V ${shader} -o ${addCompileShadersCommand_OUTPUT_DIR}/${VAR}.spv
                DEPENDS ${shader} ${addCompileShadersCommand_DEPENDS}
                COMMENT "Compiling ${shader}"
                )

        cmrc_add_resources(${addCompileShadersCommand_TARGET_NAME} WHENCE ${addCompileShadersCommand_OUTPUT_DIR} ${addCompileShadersCommand_OUTPUT_DIR}/${VAR}.spv)
    endforeach ()
endfunction()