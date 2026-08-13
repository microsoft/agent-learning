#ifndef AppVersion
  #define AppVersion "dev"
#endif

#define AppName "Agent Learning CLI"
#define AppExeName "agent-learn.exe"
#define AppPublisher "Microsoft"

[Setup]
AppId={{8D2A4479-E23B-4449-BEA5-9F846ECF08F9}
AppName={#AppName}
AppVersion={#AppVersion}
AppPublisher={#AppPublisher}
DefaultDirName={autopf}\Agent Learning
DefaultGroupName=Agent Learning
DisableProgramGroupPage=yes
PrivilegesRequired=lowest
ChangesEnvironment=yes
ArchitecturesAllowed=x64compatible
ArchitecturesInstallIn64BitMode=x64compatible
Compression=lzma
SolidCompression=yes
WizardStyle=modern
SourceDir=..\..
OutputDir=dist-installer
OutputBaseFilename=agent-learning-cli-{#AppVersion}-windows-x64
UninstallDisplayIcon={app}\{#AppExeName}

[Languages]
Name: "english"; MessagesFile: "compiler:Default.isl"

[Tasks]
Name: "addtopath"; Description: "Add agent-learn command to user PATH"; GroupDescription: "Additional tasks:"

[Files]
Source: "dist\agent-learn\*"; DestDir: "{app}"; Flags: ignoreversion recursesubdirs createallsubdirs

[Icons]
Name: "{autoprograms}\Agent Learning CLI"; Filename: "{app}\{#AppExeName}"

[Registry]
Root: HKCU; Subkey: "Environment"; ValueType: expandsz; ValueName: "Path"; ValueData: "{code:PathWithEntryFirst|{app}}"; Tasks: addtopath
Root: HKCU; Subkey: "Software\AgentLearningCLI"; ValueType: string; ValueName: "AddedToUserPath"; ValueData: "1"; Tasks: addtopath; Flags: uninsdeletekey

[Code]
function HasNonEmptyPathValue: Boolean;
var
  ExistingPath: string;
begin
  Result := RegQueryStringValue(HKCU, 'Environment', 'Path', ExistingPath) and (Trim(ExistingPath) <> '');
end;

function NormalizePathEntry(Value: string): string;
begin
  Result := Trim(Uppercase(RemoveQuotes(Value)));
  while (Length(Result) > 0) and ((Result[Length(Result)] = '\') or (Result[Length(Result)] = '/')) do
    Delete(Result, Length(Result), 1);
end;

procedure SplitPathEntries(PathValue: string; var Entries: TArrayOfString);
var
  Segment: string;
  SegmentIndex: Integer;
  SeparatorIndex: Integer;
  Working: string;
begin
  SetArrayLength(Entries, 0);
  Working := PathValue;
  while True do
  begin
    SeparatorIndex := Pos(';', Working);
    if SeparatorIndex = 0 then
    begin
      Segment := Working;
      Working := '';
    end
    else
    begin
      Segment := Copy(Working, 1, SeparatorIndex - 1);
      Delete(Working, 1, SeparatorIndex);
    end;

    Segment := Trim(Segment);
    if Segment <> '' then
    begin
      SegmentIndex := GetArrayLength(Entries);
      SetArrayLength(Entries, SegmentIndex + 1);
      Entries[SegmentIndex] := Segment;
    end;

    if SeparatorIndex = 0 then
      Break;
  end;
end;

function PathContainsEntry(OrigPath: string; Dir: string): Boolean;
var
  Entries: TArrayOfString;
  I: Integer;
begin
  Result := False;
  SplitPathEntries(OrigPath, Entries);
  for I := 0 to GetArrayLength(Entries) - 1 do
  begin
    if NormalizePathEntry(Entries[I]) = NormalizePathEntry(Dir) then
    begin
      Result := True;
      Exit;
    end;
  end;
end;

function RemovePathEntry(OrigPath: string; Dir: string): string;
var
  Entries: TArrayOfString;
  I: Integer;
  NormalizedDir: string;
begin
  Result := '';
  NormalizedDir := NormalizePathEntry(Dir);
  SplitPathEntries(OrigPath, Entries);
  for I := 0 to GetArrayLength(Entries) - 1 do
  begin
    if NormalizePathEntry(Entries[I]) = NormalizedDir then
      Continue;
    if Result = '' then
      Result := Entries[I]
    else
      Result := Result + ';' + Entries[I];
  end;
end;

function PathWithEntryFirst(Dir: string): string;
var
  OrigPath: string;
  RemainingPath: string;
begin
  Result := Dir;
  if not RegQueryStringValue(HKCU, 'Environment', 'Path', OrigPath) or (Trim(OrigPath) = '') then
    Exit;

  RemainingPath := RemovePathEntry(OrigPath, Dir);
  if RemainingPath <> '' then
    Result := Dir + ';' + RemainingPath;
end;

procedure CurUninstallStepChanged(CurUninstallStep: TUninstallStep);
var
  AddedToPath: string;
  CurrentPath: string;
  UpdatedPath: string;
begin
  if CurUninstallStep <> usUninstall then
    Exit;
  if not RegQueryStringValue(HKCU, 'Software\AgentLearningCLI', 'AddedToUserPath', AddedToPath) then
    Exit;
  if not RegQueryStringValue(HKCU, 'Environment', 'Path', CurrentPath) then
    Exit;

  UpdatedPath := RemovePathEntry(CurrentPath, ExpandConstant('{app}'));
  if UpdatedPath = CurrentPath then
    Exit;

  if UpdatedPath = '' then
    RegDeleteValue(HKCU, 'Environment', 'Path')
  else
    RegWriteExpandStringValue(HKCU, 'Environment', 'Path', UpdatedPath);
end;
