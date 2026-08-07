#ifndef AppVersion
  #define AppVersion "dev"
#endif

#define AppName "Agents Learning SDK CLI"
#define AppExeName "agent-learning.exe"
#define AppPublisher "Microsoft"

[Setup]
AppId={{8D2A4479-E23B-4449-BEA5-9F846ECF08F9}
AppName={#AppName}
AppVersion={#AppVersion}
AppPublisher={#AppPublisher}
DefaultDirName={autopf}\Agents Learning SDK
DefaultGroupName=Agents Learning SDK
DisableProgramGroupPage=yes
ChangesEnvironment=yes
ArchitecturesAllowed=x64compatible
ArchitecturesInstallIn64BitMode=x64compatible
Compression=lzma
SolidCompression=yes
WizardStyle=modern
OutputDir=dist-installer
OutputBaseFilename=agents-learning-sdk-cli-{#AppVersion}-windows-x64
UninstallDisplayIcon={app}\{#AppExeName}

[Languages]
Name: "english"; MessagesFile: "compiler:Default.isl"

[Tasks]
Name: "addtopath"; Description: "Add agent-learning command to PATH"; GroupDescription: "Additional tasks:"; Flags: checkedonce

[Files]
Source: "dist\agent-learning.exe"; DestDir: "{app}"; Flags: ignoreversion

[Icons]
Name: "{autoprograms}\Agents Learning CLI"; Filename: "{app}\{#AppExeName}"

[Registry]
Root: HKCU; Subkey: "Environment"; ValueType: expandsz; ValueName: "Path"; ValueData: "{olddata};{app}"; Tasks: addtopath; Check: NeedsAddPath(ExpandConstant('{app}'))

[Code]
function NeedsAddPath(Dir: string): Boolean;
var
  OrigPath: string;
begin
  Result := True;
  if not RegQueryStringValue(HKCU, 'Environment', 'Path', OrigPath) then
    Exit;

  if Pos(';' + Uppercase(Dir) + ';', ';' + Uppercase(OrigPath) + ';') > 0 then
    Result := False;
end;
